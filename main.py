import time
from pyrxbm.binary import BinaryRobloxFile


def main():
    import numpy as np

    root = BinaryRobloxFile()
    with open(
        "E:\\Nextcloud\\blender\\quad\\bc\\roblox\\TH_lay1.saved.rbxm", "rb"
    ) as file:
        readtime = time.perf_counter()
        root.deserialize(file)
        readtime = time.perf_counter() - readtime
    keyframes = [o for o in root.Instances if o.ClassName == "Keyframe"]
    keyframeSequences = [o for o in root.Instances if o.ClassName == "KeyframeSequence"]
    times = [k.Time for k in keyframes]
    instances_read = root.Instances[:]
    class_names_read = [c.ClassName for c in root.Classes]
    chunk_data_read = [chunk.Data for chunk in root.Chunks]
    print(root)
    with open(
        "E:\\Nextcloud\\blender\\quad\\bc\\roblox\\TH_lay1.pysaved.rbxm", "wb"
    ) as file:
        writetime = time.perf_counter()
        root.serialize(file)
        writetime = time.perf_counter() - writetime
    instances_written = root.Instances[:]
    class_names_written = [c.ClassName for c in root.Classes]
    chunk_data_written = [chunk.Data for chunk in root.Chunks]
    poses = [root.Instances[id] for id in root.ClassMap["Pose"].InstanceIds]
    components = np.row_stack([pose.CFrame.GetComponents() for pose in poses])
    rots = components[:, 3:]
    weirdpose = poses[1200]
    print(f"Read time: {readtime}; Write time: {writetime}")
    print(
        f"CFrames read   : {chunk_data_read[17][:60].hex()} ({len(chunk_data_read[17])}b)"
    )
    print(
        f"CFrames written: {chunk_data_written[22][:60].hex()} ({len(chunk_data_written[22])}b)"
    )
    print(f"CFrames equal: {chunk_data_read[17] == chunk_data_written[22]}")
    assert instances_read == instances_written
    assert class_names_read == class_names_written
    # assert chunk_data_read[1:4] == chunk_data_written


""" lua code
a = workspace.AvatarModels["Teegle Horse"].StarterCharacter.AnimSaves.TH_lay1
frames = a:GetChildren()
table.sort(frames, function(a,b) return a.Time < b.Time end)
weirdpose = frames[8].RootPart.Origin.mPelvis.mSpine1.mSpine2.mTorso.mSpine3.mSpine4.mChest.mCollarRight.mShoulderRight.mElbowRight.mWristRight
print(frames)
print(frames[1].BELLY.CFrame)
2.72848411e-12, 1.1920929e-07, -1.1920929e-07, 1, 4.54747351e-13, -4.06575815e-20, -4.54747351e-13, 1, -8.94069672e-08, 0, 8.94069672e-08, 1
print(weirdpose.CFrame:GetComponents())
     2.4586915969848633e-07 0.0000057220458984375 -0.0000025033950805664062 
     0.9999739527702332 0.007149409502744675 0.0009786344598978758 
    -0.007205313071608543 0.9966623783111572 0.08131515234708786 
    -0.00039401277899742126 -0.08132007718086243 0.9966880083084106
"""

""" c# code
{2.728484E-12, 1.192093E-07, -1.192093E-07, 1, 4.547474E-13, -4.065758E-20, -4.547474E-13, 1, -8.940697E-08, 0, 8.940697E-08, 1}
"""


if __name__ == "__main__":
    main()
