import math

if __name__ == "__main__":
    # Define the configurable parameters for the simulator

    # Size of a memory address in bits
    memory_address_bits = 16
    memory_size = 2 ** memory_address_bits  # Total memory size in bytes

    # Size of the cache in bytes (must be a power of 2)
    cache_size = 2 ** 10

    # Size of a cache block in bytes (must be a power of 2)
    cache_block_size = 2 ** 6

    # Cache associativity (must be a power of 2)
    cache_associativity = 2 ** 0

    num_blocks = cache_size // cache_block_size

    num_sets = num_blocks // cache_associativity

    offset_length = int(math.log2(cache_block_size))

    index_length = int(math.log2(num_sets))

    tag_length = memory_address_bits - (offset_length + index_length)

    # Initialize memory so that reading from address A returns A
    memory = [0] * memory_size
    for i in range(memory_size -4):
        if i % 4 == 0:
            curr = i
            if i > (256 ** 3):
                memory[i - 3] = curr // (256 ** 3)
                curr = curr % (256 ** 3)

            if i > (256 ** 2):
                memory[i - 2] = curr // (256 ** 2)
                curr = curr % (256 ** 2)

            if i > 256:
                memory[i - 1] = curr // 256
                curr = curr % 256

            memory[i] = curr


    class CacheBlock:
        def __init__(self, block_size):
            self.tag = -1  # Initialize tag to -1
            self.data = bytearray(block_size)  # Block data

    class CacheSet:
        def __init__(self, associativity, block_size):
            self.blocks = [CacheBlock(block_size) for _ in range(associativity)]

    # Initialize cache as a list of sets
    cache = [CacheSet(cache_associativity, cache_block_size) for _ in range(num_sets)]


    def decode_address(A):
        block_offset = A & ((1 << offset_length) - 1)
        index = (A >> offset_length) & ((1 << index_length) - 1)
        tag = A >> (offset_length + index_length)
        return [tag, index, block_offset]


    def read_word(A):
        [tag, index, block_offset] = decode_address(A)
        cache_set = cache[index]

        for block in cache_set.blocks:
            if block.tag == tag:
                # Cache hit: Extract the word from the block
                print("read hit" + "index: " + index + "tag: " + tag)
                word = int.from_bytes(block.data[block_offset:block_offset + 4], 'little')
                print (word)
                return word

        # Cache miss: Load block from memory
        print("read miss" + "index: " + index + "tag: " + tag)
        memory_block_start = A - block_offset
        new_block = CacheBlock(cache_block_size)
        new_block.tag = tag
        new_block.data = memory[memory_block_start : (memory_block_start + cache_block_size)]

        # Replace the first block in the set (simplest replacement policy)
        cache_set.blocks[0] = new_block

        # Return the requested word
        word = int.from_bytes(new_block.data[block_offset:block_offset + 4], 'little')
        print("word: " + word)
        return word

    # Test Case 1:
    print("---------------")
    print("cache size = " + str(cache_size))
    print("block size = " + str(cache_block_size))
    print("#blocks = " + str(num_blocks))
    print("#sets = " + str(num_sets))
    print("associativity = " + str(cache_associativity))
    print("tag length = " + str(tag_length))
    print("---------------")


    Addresses = [0, 0, 60, 64, 1000, 1028, 12920, 12924, 12928]
    for add in Addresses:
        read_word(add)