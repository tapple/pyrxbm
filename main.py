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
        times = [k.Time for k in keyframes]
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
2.72848411e-12, 1.1920929e-07, -1.1920929e-07, 1, 4.54747351e-13, -4.06575815e-20, -4.54747351e-13, 1, -8.94069672e-08, 0, 8.94069672e-08, 1
"""

""" c# code
{2.728484E-12, 1.192093E-07, -1.192093E-07, 1, 4.547474E-13, -4.065758E-20, -4.547474E-13, 1, -8.940697E-08, 0, 8.940697E-08, 1}
"""


if __name__ == "__main__":
    main()
