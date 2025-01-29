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

    # Load instructions from a file
    start_index = 100
    file_path = "instructions.txt"
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            parts = line.strip().replace(",", "").split()
            operation = parts[0]

            # Extract values, defaulting to None where components are absent
            Rd = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith('r') else None
            Rs1 = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith('r') else None
            Rs2 = int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith('r') else None
            immed = int(parts[-1]) if parts[-1].isdigit() else None

            opcode = opcode_map.get(operation, 0)
            instruction = Instruction.build_instruction(opcode, Rd, Rs1, Rs2, immed)

            cpu.memory[start_index + i] = instruction

