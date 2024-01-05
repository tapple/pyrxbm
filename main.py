from pyrxbm.binary import BinaryRobloxFile


def main():
    with open(
        "E:\\Nextcloud\\blender\\quad\\bc\\roblox\\TH_lay1.saved.rbxm", "rb"
    ) as file:
        root = BinaryRobloxFile()
        root.deserialize(file)
        keyframes = [o for o in root.Instances if o.ClassName == "Keyframe"]
        keyframeSequences = [
            o for o in root.Instances if o.ClassName == "KeyframeSequence"
        ]
        times = [k.props["Time"] for k in keyframes]
        print(root)
    # for i, chunk in enumerate(root.Chunks):
    #     with open(
    #         f"E:\\Nextcloud\\blender\\quad\\bc\\roblox\\TH_lay1\\chunk{i:02d}.lz4", "wb") as chunkfile:
    #         chunkfile.write(chunk.CompressedData)
    # chunkfile = open("E:\\Nextcloud\\blender\\quad\\bc\\roblox\\TH_lay1\\chunk00", "wb")
    # chunkfile.write(root.Chunks[0].CompressedData[2:])


""" lua code
a = workspace.AvatarModels["Teegle Horse"].StarterCharacter.AnimSaves.TH_lay1
frames = a:GetChildren()
table.sort(frames, function(a,b) return a.Time < b.Time end)
print(frames)
print(frames[1].BELLY.CFrame)
"""


if __name__ == "__main__":
    main()
