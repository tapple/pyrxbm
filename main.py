from pyrxbm.binary import BinaryRobloxFile


def main():
    with open(
        "G:\\Nextcloud\\blender\\quad\\bc\\roblox\\TH_lay1.saved.rbxm", "rb"
    ) as file:
        root = BinaryRobloxFile()
        root.deserialize(file)
        print(root)
    # for i, chunk in enumerate(root.Chunks):
    #     with open(
    #         f"E:\\Nextcloud\\blender\\quad\\bc\\roblox\\TH_lay1\\chunk{i:02d}.lz4", "wb") as chunkfile:
    #         chunkfile.write(chunk.CompressedData)
    # chunkfile = open("E:\\Nextcloud\\blender\\quad\\bc\\roblox\\TH_lay1\\chunk00", "wb")
    # chunkfile.write(root.Chunks[0].CompressedData[2:])


if __name__ == "__main__":
    main()
