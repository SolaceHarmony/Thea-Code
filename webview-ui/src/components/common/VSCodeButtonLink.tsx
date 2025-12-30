import { VSCodeButton, VSCodeButtonProps } from "../ui/vscode-components"

interface VSCodeButtonLinkProps extends VSCodeButtonProps {
        href: string
}

export const VSCodeButtonLink = ({ href, children, ...props }: VSCodeButtonLinkProps) => (
        <VSCodeButton href={href} {...props}>
                {children}
        </VSCodeButton>
)
