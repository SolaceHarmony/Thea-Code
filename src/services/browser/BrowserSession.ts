import * as vscode from "vscode"
import { Browser, Page, ScreenshotOptions, TimeoutError, connect } from "puppeteer-core"
import pWaitFor from "p-wait-for"
import delay from "delay"
import { BrowserActionResult } from "../../shared/ExtensionMessage"
import { discoverChromeHostUrl, tryChromeHostUrl } from "./browserDiscovery"
import { UrlContentFetcher } from "./UrlContentFetcher" // adjust path as needed
import { ConsoleMessage } from "puppeteer-core"
import { getErrorMessage } from "../../shared/errors"

export class BrowserSession {
	private context: vscode.ExtensionContext
	private browser?: Browser
	private page?: Page
	private currentMousePosition?: string
	private lastConnectionAttempt?: number
	private urlContentFetcher: UrlContentFetcher

	constructor(context: vscode.ExtensionContext) {
		this.context = context
		this.urlContentFetcher = new UrlContentFetcher(context)
	}

	/**
	 * Gets the viewport size from global state or returns default
	 */
	private getViewport(): { width: number; height: number } {
		const size = this.context.globalState.get<string>("browserViewportSize") ?? "900x600"
		const [width, height] = size.split("x").map(Number)
		return { width, height }
	}

	/**
	 * Launches a local browser instance
	 */
	private async launchLocalBrowser(): Promise<void> {
		console.log("Launching local browser")
		const stats = (await this.urlContentFetcher.ensureChromiumExists()) as {
			puppeteer: typeof import("puppeteer-core")
			executablePath: string
		}
		this.browser = await stats.puppeteer.launch({
			args: [
				"--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
			],
			executablePath: stats.executablePath,
			defaultViewport: this.getViewport(),
			// headless: false,
		})
	}

	/**
	 * Connects to a browser using a WebSocket URL
	 */
	private async connectWithChromeHostUrl(chromeHostUrl: string): Promise<boolean> {
		try {
			this.browser = await connect({
				browserURL: chromeHostUrl,
				defaultViewport: this.getViewport(),
			})

			// Cache the successful endpoint
			console.log(`Connected to remote browser at ${chromeHostUrl}`)
			this.context.globalState.update("cachedChromeHostUrl", chromeHostUrl)
			this.lastConnectionAttempt = Date.now()

			return true
  } catch (error) {
			console.log(`Failed to connect using WebSocket endpoint: ${getErrorMessage(error)}`)
			return false
		}
	}

	/**
	 * Attempts to connect to a remote browser using various methods
	 * Returns true if connection was successful, false otherwise
	 */
	private async connectToRemoteBrowser(): Promise<boolean> {
		const remoteBrowserHost = this.context.globalState.get<string>("remoteBrowserHost")

		// Try to connect with cached endpoint first if it exists and is recent (less than 1 hour old)
		const cachedChromeHostUrl = this.context.globalState.get<string>("cachedChromeHostUrl")
		if (cachedChromeHostUrl && this.lastConnectionAttempt && Date.now() - this.lastConnectionAttempt < 3_600_000) {
			console.log(`Attempting to connect using cached Chrome Host Url: ${cachedChromeHostUrl}`)
			if (await this.connectWithChromeHostUrl(cachedChromeHostUrl)) {
				return true
			}

			console.log(`Failed to connect using cached Chrome Host Url: ${cachedChromeHostUrl}`)
			// Clear the cached endpoint since it's no longer valid
			this.context.globalState.update("cachedChromeHostUrl", undefined)

		}

		// If user provided a remote browser host, try to connect to it
		else if (remoteBrowserHost) {
			console.log(`Attempting to connect to remote browser at ${remoteBrowserHost}`)
			try {
				const hostIsValid = await tryChromeHostUrl(remoteBrowserHost)
				
				if (hostIsValid) {
					console.log(`Found WebSocket endpoint: ${remoteBrowserHost}`)
					if (await this.connectWithChromeHostUrl(remoteBrowserHost)) {
						return true
					}
				} else {
					console.warn("Could not validate remote browser host response; skipping direct connect attempt")
				}
		   } catch (error) {
				console.error(`Failed to connect to remote browser: ${getErrorMessage(error)}`)
				// Fall back to auto-discovery if remote connection fails
			}
		}

		try {
			console.log("Attempting browser auto-discovery...")
			const chromeHostUrl = await discoverChromeHostUrl()

			if (chromeHostUrl && (await this.connectWithChromeHostUrl(chromeHostUrl))) {
				return true
			}
  } catch (error) {
			console.error(`Auto-discovery failed: ${getErrorMessage(error)}`)
			// Fall back to local browser if auto-discovery fails
		}

		return false
	}

	async launchBrowser(): Promise<void> {
		console.log("launch browser called")

		// Check if remote browser connection is enabled
		const remoteBrowserEnabled = this.context.globalState.get<boolean>("remoteBrowserEnabled")

		if (!remoteBrowserEnabled) {
			console.log("Launching local browser")
			if (this.browser) {
				// throw new Error("Browser already launched")
				await this.closeBrowser() // this may happen when the model launches a browser again after having used it already before
			} else {
				// If browser wasn't open, just reset the state
				this.resetBrowserState()
			}
			await this.launchLocalBrowser()
		} else {
			console.log("Connecting to remote browser")
			// Remote browser connection is enabled
			const remoteConnected = await this.connectToRemoteBrowser()

			// If all remote connection attempts fail, fail loudly (no implicit local fallback)
			if (!remoteConnected) {
				throw new Error("Failed to connect to remote browser; no fallback to local is performed")
			}
		}
	}

	/**
	 * Closes the browser and resets browser state
	 */
	async closeBrowser(): Promise<BrowserActionResult> {
		if (this.browser || this.page) {
			console.log("closing browser...")

			const remoteBrowserEnabled = this.context.globalState.get<boolean>("remoteBrowserEnabled")
			if (remoteBrowserEnabled && this.browser) {
				await this.browser.disconnect().catch(() => {})
			} else {
				await this.browser?.close().catch(() => {})
				this.resetBrowserState()
			}

			// this.resetBrowserState()
		}
		return {}
	}

	/**
	 * Resets all browser state variables
	 */
	private resetBrowserState(): void {
		this.browser = undefined
		this.page = undefined
		this.currentMousePosition = undefined
	}

	async doAction(action: (page: Page) => Promise<void>): Promise<BrowserActionResult> {
		if (!this.page) {
			throw new Error(
				"Browser is not launched. This may occur if the browser was automatically closed by a non-`browser_action` tool.",
			)
		}

		const logs: string[] = []
		let lastLogTs = Date.now()

		const consoleListener = (msg: ConsoleMessage) => {
			if (msg.type() === "log") {
				logs.push(msg.text())
			} else {
				logs.push(`[${msg.type()}] ${msg.text()}`)
			}
			lastLogTs = Date.now()
		}

		const errorListener = (err: Error) => {
			logs.push(`[Page Error] ${err.toString()}`)
			lastLogTs = Date.now()
		}

		// Add the listeners
		this.page.on("console", consoleListener)
		this.page.on("pageerror", errorListener)

		try {
			try {
				await action(this.page)
			} catch (err) {
				if (!(err instanceof TimeoutError)) {
					const msg = getErrorMessage(err)
					logs.push(`[Error] ${msg}`)
				}
			}

			// Wait for console inactivity, with a timeout
			await pWaitFor(() => Date.now() - lastLogTs >= 500, {
				timeout: 3_000,
				interval: 100,
			}).catch(() => {})

			// Get viewport dimensions and user preferences
			const viewport = this.getViewport()
			const screenshotPrefs = this.getScreenshotPreferences()

			let options: ScreenshotOptions = {
				encoding: "base64",
				fullPage: screenshotPrefs.fullPage,
				
				// Enable clipping if the user has enabled it in settings
				...(this.context.globalState.get<boolean>("useClipping") && {
					clip: {
						x: 0,
						y: 0,
						width: viewport.width,
						height: viewport.height,
					}
				}),
			}

			let screenshotBase64: string
			let screenshot: string | undefined

			// Capture screenshot using the configured format; no implicit fallbacks
			const result = (await this.page.screenshot({
				...options,
				type: screenshotPrefs.format,
				...(screenshotPrefs.format === "webp" && { quality: screenshotPrefs.quality }),
			})) as unknown
			const base = typeof result === "string" ? result : (result as string)
			if (base && base.length > 0) {
				screenshotBase64 = base
				screenshot = `data:image/${screenshotPrefs.format};base64,${screenshotBase64}`
			}

			if (!screenshot) {
				throw new Error("Failed to capture screenshot")
			}

			// Cap logs length to avoid oversized IPC payloads (100 KB)
			const joined = logs.join("\n")
			const maxLen = 100_000
			const finalLogs = joined.length > maxLen ? `${joined.slice(0, maxLen)}\n[logs truncated at ${maxLen} chars]` : joined

			return {
				screenshot,
				logs: finalLogs,
				currentUrl: this.page.url(),
				currentMousePosition: this.currentMousePosition,
				captureTimestamp: Date.now(),
			}
		} finally {
			// Ensure listeners are always removed even on error
			this.page.off("console", consoleListener)
			this.page.off("pageerror", errorListener)
		}
	}

	/**
	 * Extract the root domain from a URL
	 * e.g., http://localhost:3000/path -> localhost:3000
	 * e.g., https://example.com/path -> example.com
	 */
	private getRootDomain(url: string): string {
		try {
			const urlObj = new URL(url)
			// Remove www. prefix if present
			return urlObj.host.replace(/^www\./, "")
		} catch {
			// If URL parsing fails, return the original URL
			return url
		}
	}

	/**
	 * Navigate to a URL with standard loading options
	 */
	private async navigatePageToUrl(page: Page, url: string): Promise<void> {
		const navTimeout = this.context.globalState.get<number>("pageGotoTimeoutMs") ?? 7_000
		const htmlStableTimeout = this.context.globalState.get<number>("htmlStableTimeoutMs") ?? 5_000
		await page.goto(url, { timeout: navTimeout, waitUntil: ["domcontentloaded", "networkidle2"] })
		await this.waitTillHTMLStable(page, htmlStableTimeout)
	}

	/**
	 * Creates a new tab and navigates to the specified URL
	 */
	private async createNewTab(url: string): Promise<BrowserActionResult> {
		if (!this.browser) {
			throw new Error("Browser is not launched")
		}

		// Create a new page and set as the active page
		this.page = await this.browser.newPage()
		
		// Navigate to the URL
		const res = await this.doAction(async (page) => {
			await this.navigatePageToUrl(page, url)
		})
		const pageCount = (await this.browser.pages()).length
		return { ...res, createdNewTab: true, pageCount }
	}

	async navigateToUrl(url: string): Promise<BrowserActionResult> {
		if (!this.browser) {
			throw new Error("Browser is not launched")
		}
		// Remove trailing slash for comparison
		const normalizedNewUrl = url.replace(/\/$/, "")

		// Extract the root domain from the URL
		const rootDomain = this.getRootDomain(normalizedNewUrl)

		// Get all current pages
		const pages = await this.browser.pages()

		// Try to find a page with the same root domain
		let existingPage: Page | undefined

		for (const page of pages) {
			try {
				const pageUrl = page.url()
				if (pageUrl && this.getRootDomain(pageUrl) === rootDomain) {
					existingPage = page
					break
				}
			} catch (error) {
				// Skip pages that might have been closed or have errors
				console.log(`Error checking page URL: ${error}`)
			}
		}

		if (existingPage) {
			// Tab with the same root domain exists, switch to it
			console.log(`Tab with domain ${rootDomain} already exists, switching to it`)

			// Update the active page
			this.page = existingPage
			await existingPage.bringToFront()

			// Navigate to the new URL if it's different]
			const currentUrl = existingPage.url().replace(/\/$/, "") // Remove trailing / if present
			if (this.getRootDomain(currentUrl) === rootDomain && currentUrl !== normalizedNewUrl) {
				console.log(`Navigating to new URL: ${normalizedNewUrl}`)
				console.log(`Current URL: ${currentUrl}`)
				console.log(`Root domain: ${this.getRootDomain(currentUrl)}`)
				console.log(`New URL: ${normalizedNewUrl}`)
				// Navigate to the new URL
				const res = await this.doAction(async (page) => {
					await this.navigatePageToUrl(page, normalizedNewUrl)
				})
				const pageCount = (await this.browser.pages()).length
				return { ...res, reusedTab: true, pageCount }
			} else {
				console.log(`Tab with domain ${rootDomain} already exists, and URL is the same: ${normalizedNewUrl}`)
				// URL is the same, just reload the page to ensure it's up to date
				console.log(`Reloading page: ${normalizedNewUrl}`)
				console.log(`Current URL: ${currentUrl}`)
				console.log(`Root domain: ${this.getRootDomain(currentUrl)}`)
				console.log(`New URL: ${normalizedNewUrl}`)
				const res = await this.doAction(async (page) => {
					const navTimeout = this.context.globalState.get<number>("pageGotoTimeoutMs") ?? 7_000
					const htmlStableTimeout = this.context.globalState.get<number>("htmlStableTimeoutMs") ?? 5_000
					await page.reload({ timeout: navTimeout, waitUntil: ["domcontentloaded", "networkidle2"] })
					await this.waitTillHTMLStable(page, htmlStableTimeout)
				})
				const pageCount = (await this.browser.pages()).length
				return { ...res, reloaded: true, reusedTab: true, pageCount }
			}
		} else {
			// No tab with this root domain exists, create a new one
			console.log(`No tab with domain ${rootDomain} exists, creating a new one`)
			return this.createNewTab(normalizedNewUrl)
		}
	}

	// page.goto { waitUntil: "networkidle0" } may not ever resolve, and not waiting could return page content too early before js has loaded
	// https://stackoverflow.com/questions/52497252/puppeteer-wait-until-page-is-completely-loaded/61304202#61304202
	private async waitTillHTMLStable(page: Page, timeout = 5_000) {
		const checkDurationMsecs = 500 // 1000
		const maxChecks = timeout / checkDurationMsecs
		let lastHTMLSize = 0
		let checkCounts = 1
		let countStableSizeIterations = 0
		const minStableSizeIterations = 3

		while (checkCounts++ <= maxChecks) {
			let html = await page.content()
			let currentHTMLSize = html.length

			// let bodyHTMLSize = await page.evaluate(() => document.body.innerHTML.length)
			console.log("last: ", lastHTMLSize, " <> curr: ", currentHTMLSize)

			if (lastHTMLSize !== 0 && currentHTMLSize === lastHTMLSize) {
				countStableSizeIterations++
			} else {
				countStableSizeIterations = 0 //reset the counter
			}

			if (countStableSizeIterations >= minStableSizeIterations) {
				console.log("Page rendered fully...")
				break
			}

			lastHTMLSize = currentHTMLSize
			await delay(checkDurationMsecs)
		}
	}

	/**
	 * Handles mouse interaction with network activity monitoring
	 */
	private async handleMouseInteraction(
		page: Page,
		coordinate: string,
		action: (x: number, y: number) => Promise<void>,
	): Promise<void> {
		const [x, y] = coordinate.split(",").map(Number)

		// Set up network request monitoring
		let hasNetworkActivity = false
		const requestListener = () => {
			hasNetworkActivity = true
		}
		page.on("request", requestListener)

		// Perform the mouse action
		await action(x, y)
		this.currentMousePosition = coordinate

		// Small delay to check if action triggered any network activity
		await delay(100)

		if (hasNetworkActivity) {
			// If we detected network activity, wait for navigation/loading
			await page
				.waitForNavigation({
					waitUntil: ["domcontentloaded", "networkidle2"],
					timeout: 7000,
				})
				.catch(() => {})
			await this.waitTillHTMLStable(page)
		}

		// Clean up listener
		page.off("request", requestListener)
	}

	async click(coordinate: string): Promise<BrowserActionResult> {
		return this.doAction(async (page) => {
			await this.handleMouseInteraction(page, coordinate, async (x, y) => {
				await page.mouse.click(x, y)
			})
		})
	}

	async type(text: string): Promise<BrowserActionResult> {
		return this.doAction(async (page) => {
			await page.keyboard.type(text)
		})
	}

	/**
	 * Scrolls the page by the specified amount
	 */
	private async scrollPage(page: Page, direction: "up" | "down"): Promise<void> {
		const { height } = this.getViewport()
		const scrollAmount = direction === "down" ? height : -height

		await page.evaluate((scrollHeight) => {
			window.scrollBy({
				top: scrollHeight,
				behavior: "auto",
			})
		}, scrollAmount)

		await delay(300)
	}

	async scrollDown(): Promise<BrowserActionResult> {
		return this.doAction(async (page) => {
			await this.scrollPage(page, "down")
		})
	}

	async scrollUp(): Promise<BrowserActionResult> {
		return this.doAction(async (page) => {
			await this.scrollPage(page, "up")
		})
	}

	async hover(coordinate: string): Promise<BrowserActionResult> {
		return this.doAction(async (page) => {
			await this.handleMouseInteraction(page, coordinate, async (x, y) => {
				await page.mouse.move(x, y)
				// Small delay to allow any hover effects to appear
				await delay(300)
			})
		})
	}

	/**
	 * Gets screenshot preferences from user settings
	 * @returns Object containing format and quality preferences
	 */
	private getScreenshotPreferences(): { format: "webp" | "png"; quality: number; fullPage: boolean } {
		return {
			format: this.context.globalState.get<string>("screenshotFormat") === "png" ? "png" : "webp",
			quality: this.context.globalState.get<number>("screenshotQuality") ?? 75,
			fullPage: this.context.globalState.get<boolean>("captureFullPage") ?? false
		}
	}
}