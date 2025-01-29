from poetry.core.masonry.utils.include import Include
from sympy.strategies.core import switch

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
        instruction = self.memory[self.pc]
        self.next_pc += 1
        return instruction

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
                alu_result = Instruction.Rs1 + Instruction.Rs2
                self.registers[Instruction.Rd] = alu_result
            case 2: #ADDI
                alu_result = Instruction.Rs1 + Instruction.immed
                self.registers[Instruction.Rd] = alu_result
            case 3: #BEQ
                if Instruction.Rs1 == Instruction.Rs2:
                    self.next_pc = self.pc + Instruction.immed
            case 4: #JAL
                alu_result = self.pc + 1
                self.registers[Instruction.Rd] = alu_result
                self.next_pc = self.pc + Instruction.immed
            case 5: #LW
                eff_address = Instruction.Rs1 + Instruction.immed
                self.registers[Instruction.Rd] = self.memory[eff_address]
            case 6: #SW
                eff_address = Instruction.Rs2 + Instruction.immed
                self.memory[eff_address] = Instruction.Rs1

        self.pc = self.next_pc






