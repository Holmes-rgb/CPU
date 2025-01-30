import re
if __name__ == "__main__":
    from CPU import CPU
    from Instruction import Instruction
    # Create a CPU instance
    cpu = CPU()

    # # Example instruction word (4-byte integer)
    # example_instruction = 0xF1234567
    #
    # # Decode the instruction
    # decoded_instruction = cpu.ID(example_instruction)
    # print(f"Decoded Instruction: opcode={decoded_instruction.opcode}, Rd={decoded_instruction.Rd}, "
    #       f"Rs1={decoded_instruction.Rs1}, Rs2={decoded_instruction.Rs2}, immed={decoded_instruction.immed}")

    opcode_map = {
        "noop": 0x0,
        "add": 0x1,
        "addi": 0x2,
        "beq": 0x3,
        "jal": 0x4,
        "lw": 0x5,
        "sw": 0x6,
        "return": 0x7
    }

    patterns = {
        "noop": re.compile(r"^noop$"),
        "addi": re.compile(r"^addi r(\d+), r(\d+), (-?\d+)$"),
        "add": re.compile(r"^add r(\d+), r(\d+), r(\d+)$"),
        "beq": re.compile(r"^beq r(\d+), r(\d+), (-?\d+)$"),
        "jal": re.compile(r"^jal r(\d+), (-?\d+)$"),
        "sw": re.compile(r"^sw r(\d+), (-?\d+)\(r(\d+)\)$"),
        "lw": re.compile(r"^lw r(\d+), (-?\d+)\(r(\d+)\)$"),
        "return": re.compile(r"^return$"),
    }

    start_index = 100
    with open("instructions.txt", 'r') as file:
        for i, line in enumerate(file):
            line = line.split("#")[0].strip()
            for op, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    opcode = opcode_map[op]
                    args = [int(x) for x in match.groups()] if match.groups() else []
                    Rd = None
                    Rs1 = None
                    Rs2 = None
                    immed = None

                    # Assign based on extracted values, defaulting missing values to None
                    match op:
                        case "addi":
                            Rd = args[0]
                            Rs1 = args[1]
                            immed = args[2]
                        case "add":
                            Rd = args[0]
                            Rs1 = args[1]
                            Rs2 = args[2]
                        case "beq":
                            Rs1 = args[0]
                            Rs2 = args[1]
                            immed = args[2]
                        case "jal":
                            Rd = args[0]
                            immed = args[1]
                        case "sw":
                            Rs1 = args[0]
                            immed = args[1]
                            Rs2 = args[2]
                        case "lw":
                            Rd = args[0]
                            immed = args[1]
                            Rs1 = args[2]

                    instruction = Instruction.build_instruction(opcode, Rd, Rs1, Rs2, immed)
                    cpu.memory[cpu.pc + i] = instruction
                    break


    cpu.pc = 100
    while True:
        instructionWord = cpu.IF()
        instruction = cpu.ID(instructionWord)
        cpu.EX(instruction)
        if instruction.opcode == 7:
            break