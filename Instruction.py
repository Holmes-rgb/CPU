class Instruction:
    def __init__(self, opcode, Rd, Rs1, Rs2, immed):
        # Initialize instruction fields
        self.opcode = opcode  # Operation code
        self.Rd = Rd          # Destination register
        self.Rs1 = Rs1        # Source register 1
        self.Rs2 = Rs2        # Source register 2
        self.immed = immed    # Immediate value

    def build_instruction(opcode, Rd, Rs1, Rs2, immed):
        instr = opcode << 28
        if Rd is not None:
            instr = instr + (Rd << 24)
        if Rs1 is not None:
            instr = instr + (Rs1 << 20)
        if Rs2 is not None:
            instr = instr + (Rs2 << 16)
        if immed is not None:
            instr = instr + immed
        return instr

