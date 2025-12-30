import React, { useCallback, forwardRef } from "react"

import {
        VSCodeButton as ToolkitButton,
        VSCodeCheckbox as ToolkitCheckbox,
        VSCodeDropdown as ToolkitDropdown,
        VSCodeLink as ToolkitLink,
        VSCodeOption as ToolkitOption,
        VSCodePanelTab as ToolkitPanelTab,
        VSCodePanelView as ToolkitPanelView,
        VSCodePanels as ToolkitPanels,
        VSCodeRadio as ToolkitRadio,
        VSCodeRadioGroup as ToolkitRadioGroup,
        VSCodeTextArea as ToolkitTextArea,
        VSCodeTextField as ToolkitTextField,
} from "@vscode/webview-ui-toolkit/react"
import type { Checkbox } from "@vscode/webview-ui-toolkit"

type ToolkitButtonProps = React.ComponentProps<typeof ToolkitButton>

export interface VSCodeButtonProps extends Omit<ToolkitButtonProps, "onClick"> {
        onClick?: (event: React.MouseEvent<HTMLButtonElement | HTMLAnchorElement>) => void
}

export const VSCodeButton: React.FC<VSCodeButtonProps> = ({ children, onClick, ...props }) => {
        const handleClick = useCallback(
                (event: MouseEvent) => {
                        onClick?.(event as unknown as React.MouseEvent<HTMLButtonElement | HTMLAnchorElement>)
                },
                [onClick],
        )

        return (
                <ToolkitButton {...props} onClick={handleClick as ToolkitButtonProps["onClick"]}>
                        {children}
                </ToolkitButton>
        )
}

type ToolkitCheckboxProps = React.ComponentProps<typeof ToolkitCheckbox>

export interface VSCodeCheckboxProps extends Omit<ToolkitCheckboxProps, "onChange"> {
        onChange?: (checked: boolean) => void
}

export const VSCodeCheckbox: React.FC<VSCodeCheckboxProps> = ({ children, onChange, ...props }) => {
        const handleChange = useCallback(
                (event: Event) => {
                        if (!onChange) {
                                return
                        }

                        const target = event.target as Checkbox | null
                        onChange(Boolean(target?.checked))
                },
                [onChange],
        )

        return (
                <ToolkitCheckbox {...props} onChange={handleChange as ToolkitCheckboxProps["onChange"]}>
                        {children}
                </ToolkitCheckbox>
        )
}

type ToolkitTextFieldProps = React.ComponentProps<typeof ToolkitTextField>
type ToolkitTextFieldElement = React.ElementRef<typeof ToolkitTextField>

export interface VSCodeTextFieldProps extends Omit<ToolkitTextFieldProps, "onInput" | "onChange"> {
        onInput?: (event: React.ChangeEvent<HTMLInputElement>) => void
        onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void
}

export const VSCodeTextField = forwardRef<ToolkitTextFieldElement, VSCodeTextFieldProps>(
        ({ children, onInput, onChange, ...props }, ref) => {
                const handleInput = useCallback(
                        (event: InputEvent) => {
                                onInput?.(event as unknown as React.ChangeEvent<HTMLInputElement>)
                        },
                        [onInput],
                )

                const handleChange = useCallback(
                        (event: Event) => {
                                onChange?.(event as unknown as React.ChangeEvent<HTMLInputElement>)
                        },
                        [onChange],
                )

                return (
                        <ToolkitTextField
                                {...props}
                                ref={ref}
                                onInput={handleInput as ToolkitTextFieldProps["onInput"]}
                                onChange={handleChange as ToolkitTextFieldProps["onChange"]}>
                                {children}
                        </ToolkitTextField>
                )
        },
)

type ToolkitLinkProps = React.ComponentProps<typeof ToolkitLink>

export interface VSCodeLinkProps extends Omit<ToolkitLinkProps, "onClick"> {
        onClick?: (event: React.MouseEvent<HTMLAnchorElement>) => void
}

export const VSCodeLink: React.FC<VSCodeLinkProps> = ({ children, onClick, ...props }) => {
        const handleClick = useCallback(
                (event: MouseEvent) => {
                        onClick?.(event as unknown as React.MouseEvent<HTMLAnchorElement>)
                },
                [onClick],
        )

        return (
                <ToolkitLink {...props} onClick={handleClick as ToolkitLinkProps["onClick"]}>
                        {children}
                </ToolkitLink>
        )
}

type ToolkitTextAreaProps = React.ComponentProps<typeof ToolkitTextArea>

export interface VSCodeTextAreaProps extends Omit<ToolkitTextAreaProps, "onInput" | "onChange"> {
        onInput?: (event: React.ChangeEvent<HTMLTextAreaElement>) => void
        onChange?: (event: React.ChangeEvent<HTMLTextAreaElement>) => void
}

export const VSCodeTextArea: React.FC<VSCodeTextAreaProps> = ({ children, onInput, onChange, ...props }) => {
        const handleInput = useCallback(
                (event: InputEvent) => {
                        onInput?.(event as unknown as React.ChangeEvent<HTMLTextAreaElement>)
                },
                [onInput],
        )

        const handleChange = useCallback(
                (event: Event) => {
                        onChange?.(event as unknown as React.ChangeEvent<HTMLTextAreaElement>)
                },
                [onChange],
        )

        return (
                <ToolkitTextArea
                        {...props}
                        onInput={handleInput as ToolkitTextAreaProps["onInput"]}
                        onChange={handleChange as ToolkitTextAreaProps["onChange"]}>
                        {children}
                </ToolkitTextArea>
        )
}

type ToolkitDropdownProps = React.ComponentProps<typeof ToolkitDropdown>

export interface VSCodeDropdownProps extends Omit<ToolkitDropdownProps, "onChange"> {
        onChange?: (event: React.ChangeEvent<HTMLSelectElement>) => void
}

export const VSCodeDropdown: React.FC<VSCodeDropdownProps> = ({ children, onChange, ...props }) => {
        const handleChange = useCallback(
                (event: Event) => {
                        onChange?.(event as unknown as React.ChangeEvent<HTMLSelectElement>)
                },
                [onChange],
        )

        return (
                <ToolkitDropdown {...props} onChange={handleChange as ToolkitDropdownProps["onChange"]}>
                        {children}
                </ToolkitDropdown>
        )
}

export type VSCodeOptionProps = React.ComponentProps<typeof ToolkitOption>

export const VSCodeOption: React.FC<VSCodeOptionProps> = ({ children, ...props }) => (
        <ToolkitOption {...props}>{children}</ToolkitOption>
)

type ToolkitRadioProps = React.ComponentProps<typeof ToolkitRadio>

export interface VSCodeRadioProps extends Omit<ToolkitRadioProps, "onChange"> {
        onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void
}

export const VSCodeRadio: React.FC<VSCodeRadioProps> = ({ children, onChange, ...props }) => {
        const handleChange = useCallback(
                (event: Event) => {
                        onChange?.(event as unknown as React.ChangeEvent<HTMLInputElement>)
                },
                [onChange],
        )

        return (
                <ToolkitRadio {...props} onChange={handleChange as ToolkitRadioProps["onChange"]}>
                        {children}
                </ToolkitRadio>
        )
}

type ToolkitRadioGroupProps = React.ComponentProps<typeof ToolkitRadioGroup>

export interface VSCodeRadioGroupProps extends Omit<ToolkitRadioGroupProps, "onChange"> {
        onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void
}

export const VSCodeRadioGroup: React.FC<VSCodeRadioGroupProps> = ({ children, onChange, ...props }) => {
        const handleChange = useCallback(
                (event: Event) => {
                        onChange?.(event as unknown as React.ChangeEvent<HTMLInputElement>)
                },
                [onChange],
        )

        return (
                <ToolkitRadioGroup {...props} onChange={handleChange as ToolkitRadioGroupProps["onChange"]}>
                        {children}
                </ToolkitRadioGroup>
        )
}

export type VSCodePanelsProps = React.ComponentProps<typeof ToolkitPanels>

export const VSCodePanels: React.FC<VSCodePanelsProps> = ({ children, ...props }) => (
        <ToolkitPanels {...props}>{children}</ToolkitPanels>
)

export type VSCodePanelTabProps = React.ComponentProps<typeof ToolkitPanelTab>

export const VSCodePanelTab: React.FC<VSCodePanelTabProps> = ({ children, ...props }) => (
        <ToolkitPanelTab {...props}>{children}</ToolkitPanelTab>
)

export type VSCodePanelViewProps = React.ComponentProps<typeof ToolkitPanelView>

export const VSCodePanelView: React.FC<VSCodePanelViewProps> = ({ children, ...props }) => (
        <ToolkitPanelView {...props}>{children}</ToolkitPanelView>
)
