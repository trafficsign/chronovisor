import Darwin
import Foundation
import ServiceManagement

private let launchAgentDirectory = Bundle.main.bundleURL
    .appendingPathComponent("Contents/Library/LaunchAgents", isDirectory: true)

private func plistNames() throws -> [String] {
    try FileManager.default.contentsOfDirectory(
        at: launchAgentDirectory,
        includingPropertiesForKeys: nil
    )
    .map(\.lastPathComponent)
    .filter { $0.hasSuffix(".plist") }
    .sorted()
}

private func statusName(_ status: SMAppService.Status) -> String {
    switch status {
    case .notRegistered: return "not_registered"
    case .enabled: return "enabled"
    case .requiresApproval: return "requires_approval"
    case .notFound: return "not_found"
    @unknown default: return "unknown_\(status.rawValue)"
    }
}

private func writeReport(_ rows: [[String: Any]]) {
    let data = try! JSONSerialization.data(
        withJSONObject: ["services": rows],
        options: [.prettyPrinted, .sortedKeys]
    )
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

private func manage(_ action: String, plistName: String? = nil) -> Never {
    var rows: [[String: Any]] = []
    var failed = false
    do {
        let names = try plistNames()
        if let plistName, !names.contains(plistName) {
            throw CocoaError(.fileNoSuchFile)
        }
        for name in plistName.map({ [$0] }) ?? names {
            let service = SMAppService.agent(plistName: name)
            var errorText: String?
            do {
                if action == "register"
                    && (service.status == .notRegistered || service.status == .notFound)
                {
                    try service.register()
                } else if action == "unregister"
                    && service.status != .notRegistered
                    && service.status != .notFound
                {
                    try service.unregister()
                }
            } catch {
                errorText = String(describing: error)
            }
            let status = statusName(service.status)
            var row: [String: Any] = ["plist": name, "status": status]
            if let errorText { row["error"] = errorText }
            if errorText != nil || (action == "register" && status != "enabled") {
                failed = true
            }
            rows.append(row)
        }
    } catch {
        rows.append(["status": "error", "error": String(describing: error)])
        failed = true
    }
    writeReport(rows)
    exit(failed ? 1 : 0)
}

private func runService() -> Never {
    let command = Array(CommandLine.arguments.dropFirst(2))
    guard let executable = command.first, executable.hasPrefix("/") else {
        fputs("Chronovisor service executable must be absolute\n", stderr)
        exit(64)
    }
    var arguments: [UnsafeMutablePointer<CChar>?] = command.map { strdup($0) }
    arguments.append(nil)
    defer { arguments.dropLast().forEach { free($0) } }
    execv(executable, &arguments)
    perror("execv")
    exit(127)
}

guard CommandLine.arguments.count >= 2 else {
    fputs("usage: Chronovisor {register|unregister|status|register-one PLIST|unregister-one PLIST|open-settings|run}\n", stderr)
    exit(64)
}

switch CommandLine.arguments[1] {
case "register": manage("register")
case "unregister": manage("unregister")
case "status": manage("status")
case "register-one":
    guard CommandLine.arguments.count == 3 else { exit(64) }
    manage("register", plistName: CommandLine.arguments[2])
case "unregister-one":
    guard CommandLine.arguments.count == 3 else { exit(64) }
    manage("unregister", plistName: CommandLine.arguments[2])
case "open-settings":
    SMAppService.openSystemSettingsLoginItems()
    exit(0)
case "run": runService()
default:
    fputs("unknown command\n", stderr)
    exit(64)
}
