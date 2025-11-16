// Mock default shell based on platform
import os from "os"

let defaultShell
if (os.platform() === "win32") {
	defaultShell = "cmd.exe"
} else {
	defaultShell = "/bin/bash"
}

export default defaultShell
