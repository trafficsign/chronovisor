import Darwin

let command = Array(CommandLine.arguments.dropFirst(2))
guard CommandLine.arguments.dropFirst().first == "run",
      let executable = command.first,
      executable.hasPrefix("/") else {
    fputs("Chronovisor service executable must be absolute\n", stderr)
    exit(64)
}

var arguments: [UnsafeMutablePointer<CChar>?] = command.map { strdup($0) }
arguments.append(nil)
defer { arguments.dropLast().forEach { free($0) } }
execv(executable, &arguments)
perror("execv")
exit(127)
