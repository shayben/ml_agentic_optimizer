<#
.SYNOPSIS
Reads the local GitHub Copilot CLI OAuth token from Windows Credential Manager.

.DESCRIPTION
This helper is retained for legacy experiments that intentionally run Copilot CLI on remote compute. In the
primary MCP broker topology, Copilot CLI authenticates and runs locally; AML nodes only need CONTROL_PLANE_URL,
CONTROL_PLANE_TOKEN, and CONTROL_PLANE_RUN_ID.

When Copilot CLI runs headlessly, it reads GitHub auth from COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN. On
Windows, an interactive Copilot CLI login stores a GitHub Copilot CLI app OAuth token in Windows Credential
Manager under a target such as:

  copilot-cli/https://github.com:<github-login>

This script reads that credential via the Win32 CredReadW API and can optionally write it to Azure Key Vault.

SECURITY WARNING:
Forwarding a user OAuth token to remote infrastructure is credential movement. Prefer the local MCP broker flow;
if you intentionally use the legacy remote-Copilot path, store secrets in Key Vault/AML secrets, avoid plaintext
files and command history, limit who can read the secret, and confirm enterprise policy.
#>

[CmdletBinding()]
param(
    [string]$User,
    [string]$Target,
    [switch]$Show,
    [string]$KeyVault,
    [string]$SecretName
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-GitHubUser {
    if ($User) { return $User }

    foreach ($name in @('GITHUB_USER', 'GH_USER')) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if (-not [string]::IsNullOrWhiteSpace($value)) { return $value }
    }

    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if ($gh) {
        try {
            $login = (& gh api user --jq .login 2>$null)
            if (-not [string]::IsNullOrWhiteSpace($login)) { return $login.Trim() }
        } catch {
            # Ignore and fall through to explicit guidance below.
        }
    }

    return $null
}

if (-not $Target) {
    $resolvedUser = Resolve-GitHubUser
    if (-not $resolvedUser) {
        throw "Could not determine the GitHub login. Re-run with -User <github-login> or -Target 'copilot-cli/https://github.com:<github-login>'."
    }
    $Target = "copilot-cli/https://github.com:$resolvedUser"
}

if ($KeyVault -and -not $SecretName) {
    throw "When -KeyVault is provided, also provide -SecretName <name>."
}
if ($SecretName -and -not $KeyVault) {
    throw "When -SecretName is provided, also provide -KeyVault <vault-name>."
}

$typeName = 'Win32CredRead'
if (-not ([System.Management.Automation.PSTypeName]$typeName).Type) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class Win32CredRead
{
    public const int CRED_TYPE_GENERIC = 1;

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct CREDENTIAL
    {
        public UInt32 Flags;
        public UInt32 Type;
        public IntPtr TargetName;
        public IntPtr Comment;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
        public UInt32 CredentialBlobSize;
        public IntPtr CredentialBlob;
        public UInt32 Persist;
        public UInt32 AttributeCount;
        public IntPtr Attributes;
        public IntPtr TargetAlias;
        public IntPtr UserName;
    }

    [DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CredRead(string target, int type, int reservedFlag, out IntPtr credentialPtr);

    [DllImport("advapi32.dll", EntryPoint = "CredFree", SetLastError = true)]
    public static extern void CredFree(IntPtr buffer);
}
"@
}

$credentialPtr = [IntPtr]::Zero
try {
    $ok = [Win32CredRead]::CredRead($Target, [Win32CredRead]::CRED_TYPE_GENERIC, 0, [ref]$credentialPtr)
    if (-not $ok) {
        $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        $message = (New-Object ComponentModel.Win32Exception($code)).Message
        throw "CredReadW failed for target '$Target' (Win32 error ${code}: $message). Confirm Copilot CLI is logged in locally and use -User or -Target if the credential target differs."
    }

    $credential = [Runtime.InteropServices.Marshal]::PtrToStructure($credentialPtr, [type][Win32CredRead+CREDENTIAL])
    if ($credential.CredentialBlobSize -eq 0 -or $credential.CredentialBlob -eq [IntPtr]::Zero) {
        throw "Credential '$Target' exists but contains an empty credential blob."
    }

    $bytes = New-Object byte[] $credential.CredentialBlobSize
    [Runtime.InteropServices.Marshal]::Copy($credential.CredentialBlob, $bytes, 0, $bytes.Length)
    $token = [Text.Encoding]::UTF8.GetString($bytes).TrimEnd([char]0)

    if (-not $token.StartsWith('gho_')) {
        Write-Warning "Credential was read, but it does not start with 'gho_'. Verify this is a GitHub Copilot CLI app OAuth token before forwarding it."
    }

    if ($Show) {
        Write-Warning "SECURITY WARNING: printing a bearer token to the console can leak it into logs, scrollback, or command capture."
        Write-Output $token
    } else {
        $prefixLength = [Math]::Min(8, $token.Length)
        $prefix = $token.Substring(0, $prefixLength)
        Write-Output "Read token from target '$Target': $prefix... (length: $($token.Length)). Use -Show only if you intentionally need the raw token for legacy remote-Copilot use."
    }

    if ($KeyVault) {
        $az = Get-Command az -ErrorAction SilentlyContinue
        if (-not $az) {
            throw "Azure CLI 'az' is required for -KeyVault but was not found on PATH. Install/sign in to Azure CLI, or omit -KeyVault."
        }

        Write-Output "Writing token to Azure Key Vault '$KeyVault' secret '$SecretName'..."
        $null = (& az keyvault secret set --vault-name $KeyVault --name $SecretName --value $token --only-show-errors)
        if ($LASTEXITCODE -ne 0) {
            throw "az keyvault secret set failed with exit code $LASTEXITCODE."
        }

        Write-Output "Stored secret for legacy remote-Copilot use. The primary MCP broker flow does not require forwarding this token to AML."
    }
} finally {
    if ($credentialPtr -ne [IntPtr]::Zero) {
        [Win32CredRead]::CredFree($credentialPtr)
    }
}
