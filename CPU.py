from Instruction import Instruction
NUM_REGISTERS = 16
MEM_SIZE = 65536

class CPU:
    def __init__(self):
        # Initialize program counter and next program counter
        self.pc = 0
        self.next_pc = 0
        # Initialize memory and registers with zeros
        self.memory = [0] * MEM_SIZE
        self.registers = [0] * NUM_REGISTERS

    def IF (self):
        instruction_word = self.memory[self.pc]
        self.next_pc += 1
        return instruction_word

    def ID (self, instruction_word):
        opcode = (instruction_word >> 28) & 0xF  # Bits 28-31
        Rd = (instruction_word >> 24) & 0xF      # Bits 24-27
        Rs1 = (instruction_word >> 20) & 0xF     # Bits 20-23
        Rs2 = (instruction_word >> 16) & 0xF     # Bits 16-19
        immed = instruction_word & 0xFFFF        # Bits 0-15

        return Instruction(opcode, Rd, Rs1, Rs2, immed)

    def EX (self, Instruction):
        match Instruction.opcode:
            case 1: #ADD
                alu_result = self.registers[Instruction.Rs1] + self.registers[Instruction.Rs2]
                if Instruction.Rd != 0:
                    self.registers[Instruction.Rd] = alu_result
            case 2: #ADDI
                alu_result = self.registers[Instruction.Rs1] + Instruction.immed
                if Instruction.Rd != 0:
                    self.registers[Instruction.Rd] = alu_result
            case 3: #BEQ
                if self.registers[Instruction.Rs1] == self.registers[Instruction.Rs2]:
                    self.next_pc = self.pc + Instruction.immed
            case 4: #JAL
                alu_result = self.pc + 1
                if Instruction.Rd != 0:
                    self.registers[Instruction.Rd] = alu_result
                self.next_pc = self.pc + Instruction.immed
            case 5: #LW
                eff_address = self.registers[Instruction.Rs1] + Instruction.immed
                if Instruction.Rd != 0:
                    self.registers[Instruction.Rd] = self.memory[eff_address]
            case 6: #SW TODO: should put 7 but is putting 2
                eff_address = self.registers[Instruction.Rs2] + Instruction.immed
                self.memory[eff_address] = Instruction.Rs1
            case 7:
                for i in range(NUM_REGISTERS):
                    print("R" + str(i) + "=" + str(self.registers[i]) )

        self.pc = self.next_pc


