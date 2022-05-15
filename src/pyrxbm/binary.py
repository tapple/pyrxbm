import struct

# http://stackoverflow.com/questions/442188/readint-readbyte-readstring-etc-in-python
class BinaryStream:
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def read_bytes(self, length) -> bytes:
        return self.base_stream.read(length)

    def write_bytes(self, value):
        self.base_stream.write(value)

    def unpack(self, fmt):
        return struct.unpack(fmt, self.read_bytes(struct.calcsize(fmt)))

    def pack(self, fmt, *data):
        return self.write_bytes(struct.pack(fmt, *data))

    def read_string(self):
        length = self.unpack("<i")
        return self.read_bytes(length).decode("utf8")

    def write_string(self, s):
        self.pack("<i", len(s)
        self.write_bytes(s.encode("utf8"))

    def read_instance_ids(self, count) -> list[int]
        """ Reads and accumulates an interleaved buffer of integers. """
        values = self.unpack("i" * count)
        for i in range(1, count):
            values[i] += values[i - 1];
        return values.ToList();

    def write_instance_ids(self, values):
        """ Accumulatively writes an interleaved array of integers. """
        inst_ids = list(values)
        for i in range(1, len(inst_ids)):
            inst_ids[i] -= values[i - 1]
        self.pack("i" * len(inst_ids), *inst_ids)

    # http://stackoverflow.com/questions/32774910/clean-way-to-read-a-null-terminated-c-style-string-from-a-file
    def readCString(self):
        buf = bytearray()
        while True:
            b = self.base_stream.read(1)
            if b is None or b == b'\0':
                return buf
            else:
                buf.extend(b)

    def writeCString(self, string):
        self.write_bytes(string)
        self.write_bytes(b"\0")



class INST:


    def __init__(self):
        self.ClassIndex = 0
        self.ClassName = ""
        self.IsService = False
        self.RootedServices = []
        self.NumInstances = 0
        self.InstanceIds = []

    def deserialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        (self.ClassIndex,) = stream.unpack("i")
        self.ClassName = stream.read_string()
        self.IsService, self.NumInstances = stream.unpack("bi")
        self.InstanceIds = stream.read_instance_ids(self.NumInstances)
        file.Classes[self.ClassIndex] = self

        # Type instType = Type.GetType($"RobloxFiles.{ClassName}");
        # if instType is None:
        #     RobloxFile.LogError($"INST - Unknown class: {ClassName} while reading INST chunk.");
        #     return;

        if self.IsService:
            self.RootedServices = []
            for  i in range(self. NumInstances):
                isRooted = stream.unpack("b")
                self.RootedServices.append(isRooted);

        for i in range(self.NumInstances):
            instId = self.InstanceIds[i]
            # inst = Activator.CreateInstance(instType) as Instance;
            # inst.Referent = instId.ToString();
            # inst.IsService = IsService;
            if self.IsService:
                isRooted = self.RootedServices[i];
                # inst.Parent = file if isRooted else None
            # file.Instances[instId] = inst;

        def serialize(self, stream: BinaryStream, file: BinaryRobloxFile):
            stream.pack("i", self.ClassIndex)
            stream.write_string(self.ClassName)
            stream.pack("bi", self.IsService, self.NumInstances)
            stream.write_instance_ids(self.InstanceIds)
            if (self.IsService):
                for instId in InstanceIds:
                # Instance service = file.Instances[instId];
                # writer.Write(service.Parent == file);
                stream.pack("b", false)

        def dump(self):
            print(f"- ClassIndex:   {self.ClassIndex}");
            print(f"- ClassName:    {self.ClassName}");
            print(f"- IsService:    {self.IsService}");

            if (self.IsService and self.RootedServices is not None):
                print(f"- RootedServices: `{', '.join(self.RootedServices)}`");

            print(f"- NumInstances: {self.NumInstances}");
            print(f"- InstanceIds: `{', '.join(self.InstanceIds)}`");

class BinaryRobloxFileChunk:
    """
    BinaryRobloxFileChunk represents a generic LZ4 - compressed chunk
    of data in Roblox's Binary File Format.
    """

    def __init__(self):
        self.ChunkType = b""
        self.Reserved = -1
        self.CompressedSize = -1
        self.Size = -1
        self.CompressedData = b""
        self.Data = b""
        self.HasWriteBuffer = False
        self.WriteBuffer = bytearray()
        self.Handler = None

    @property
    def HasCompressedData(self):
        return (self.CompressedSize > 0);

    def __str__(self):
        chunkType = self.ChunkType.replace(b'\0', b' ');
        return f"'{chunkType}' Chunk ({self.Size} bytes) [{self.Handler}]";

    def deserialize(self, stream: BinaryStream):
        (self.ClassIndex,) = stream.unpack("i")
        self.ClassName = stream.read_string()
        self.IsService, self.NumInstances = stream.unpack("bi")
        self.InstanceIds = stream.read_instance_ids(self.NumInstances)
        file.Classes[self.ClassIndex] = self

    def deserialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        self.ChunkType, self. CompressedSize, self.Size, self.Reserved = stream.unpack("4siii")

        if (self.HasCompressedData):
            self.CompressedData = stream.read_bytes(self.CompressedSize);
            # Data = LZ4Codec.Decode(CompressedData, 0, CompressedSize, Size);
        else:
            self.Data = stream.read_bytes(self.Size);

"""
    public
    BinaryRobloxFileChunk(BinaryRobloxFileWriter
    writer, bool
    compress = true)
    {
    if (!writer.WritingChunk)
        throw
        new
        Exception(
            "BinaryRobloxFileChunk: Supplied writer must have WritingChunk set to true.");

    Stream
    stream = writer.BaseStream;

    using(BinaryReader
    reader = new
    BinaryReader(stream, Encoding.UTF8, true))
    {
        long
    length = (stream.Position - writer.ChunkStart);
    stream.Position = writer.ChunkStart;

    Size = (int)
    length;
    Data = reader.ReadBytes(Size);
    }

    CompressedData = LZ4Codec.Encode(Data, 0, Size);
    CompressedSize = CompressedData.Length;

    if (!compress | | CompressedSize > Size)
        {
            CompressedSize = 0;
        CompressedData = Array.Empty < byte > ();
        }

        ChunkType = writer.ChunkType;
        Reserved = 0;
        }

        public
        void
        WriteChunk(BinaryRobloxFileWriter
        writer)
        {
        // Record
        where
        we
        are
        when
        we
        start
        writing.
        var
        stream = writer.BaseStream;
        long
        startPos = stream.Position;

        // Write
        the
        chunk
        's data.
        writer.WriteString(ChunkType, true);

        writer.Write(CompressedSize);
        writer.Write(Size);

        writer.Write(Reserved);

        if (CompressedSize > 0)
            writer.Write(CompressedData);
        else
            writer.Write(Data);

        // Capture
        the
        data
        we
        wrote
        into
        a
        byte[]
        array.
        long
        endPos = stream.Position;
        int
        length = (int)(endPos - startPos);

        using(MemoryStream
        buffer = new
        MemoryStream())
        {
            stream.Position = startPos;
        stream.CopyTo(buffer, length);

        WriteBuffer = buffer.ToArray();
        HasWriteBuffer = true;
        }
        }
        }
        }
"""

class BinaryRobloxFile(RobloxFile):
    # Header Specific
    MAGIC_HEADER = b"<roblox!\x89\xff\x0d\x0a\x1a\x0a";

    def __init__(self):
        # Header Specific
        self. Version: ushort = 0
        self. NumClasses: uint = 0
        self. NumInstances: uint = 0
        self. Reserved: long = 0

        # Runtime Specific
        self.ChunksImpl: list[BinaryRobloxFileChunk] ChunksImpl = []

        self.Instances: list[Instance] = []
        self.Classes: list[INST] = []

        self.META = None
        self.SSTR = None
        self.SIGN = None

        self.Name = "Bin:";
        self.Referent = "-1";
        self.ParentLocked = true;

    @property
    def Chunks(self):
        return self.ChunksImpl

    @property
    def HasMetadata(self):
        return self.META is not None

    @property
    def Metadata(self):
        return self.META.Data if self.META else {}

    @property
    def HasSharedStrings(self):
        return self.SSTR is not None

    @property
    def SharedStrings(self):
        return self.SSTR.Strings if self.SSTR else {}


    @property
    def HasSignatures(self):
        return self.SIGN is not None

    @property
    def Signatures(self):
        return self.SIGN.Signatures if self.SIGN else []

    def deserialize(self, file):
        stream = BinaryStream(file)
        (self.version, self.subVersion, self.priority,
         self.duration) = stream.unpack("HHif")
        self.emote = stream.readCString().decode('ascii')
        (self._loop_start, self._loop_end, self.loop, self.easeIn,
         self.easeOut, self.handPosture, jointCount) = stream.unpack(
            "ffiffii")
        self.joints = list()
        for jointNum in range(jointCount):
            joint = JointMotion()
            self.joints.append(joint)
            joint.deserialize(stream)
        (constraintCount,) = stream.unpack("i")
        self.constraints = list()
        for constraintNum in range(constraintCount):
            constraint = JointConstraintSharedData()
            self.constraints.append(constraint)
            constraint.deserialize(stream)
        return self

    def deserialize(self, file):
        stream = BinaryStream(file)
        # Verify the signature of the file.
        ignature = stream.read_bytes(14);
        if (signature != self.MAGIC_HEADER):
            raise ValueError("Provided file's signature does not match BinaryRobloxFile.MAGIC_HEADER!");

        # Read header data.
        self.Version, self.NumClasses, self.NumInstances, self.Reserved = stream.unpack("HIIq")

        # Begin reading the file chunks.
        reading = true;

        self.Classes = (None,) * self.NumClasses
        self.Instances = (None,) * self.NumInstances

        while (reading):
            chunk = BinaryRobloxFileChunk()
            chunk.deserialize(stream)
            handler = null;
            if chunk.ChunkType == b"INST":
                handler = INST();
            elif chunk.ChunkType == b"PROP":
                handler = PROP();
            elif chunk.ChunkType == b"PRNT":
                handler = PRNT();
            elif chunk.ChunkType == b"META":
                handler = META();
            elif chunk.ChunkType == b"SSTR":
                handler = SSTR();
            elif chunk.ChunkType == b"SIGN":
                handler = SIGN();
            elif chunk.ChunkType == b"END\0":
                self.ChunksImpl.append(chunk);
                reading = false;
            else:
                self.LogError(f"BinaryRobloxFile - Unhandled chunk-type: {chunk.ChunkType}!");

            if handler:
                chunk_stream = BinaryStream(BytesIO(chunk.Data))
                chunk.Handler = handler;
                handler.deserialize(chunk_stream, self);
                ChunksImpl.Add(chunk);

"""
public
override
void
Save(Stream
stream)
{
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //
// Generate
the
chunk
data.
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //

using(var
workBuffer = new
MemoryStream())
using(var
writer = new
BinaryRobloxFileWriter(this, workBuffer))
{
// Clear
the
existing
data. \
    Referent = "-1";
ChunksImpl.Clear();

NumInstances = 0;
NumClasses = 0;
SSTR = null;

// Recursively
capture
all
instances and classes. \
    writer.RecordInstances(Children);

// Apply
the
recorded
instances and classes. \
    writer.ApplyClassMap();

// Write
the
INST
chunks.
    foreach(INST
inst in Classes)
writer.SaveChunk(inst);

// Write
the
PROP
chunks.
    foreach(INST
inst in Classes)
{
    var
props = PROP.CollectProperties(writer, inst);

foreach(string
propName in props.Keys)
{
    PROP
prop = props[propName];
writer.SaveChunk(prop);
}
}

// Write
the
PRNT
chunk.
    var
parents = new
PRNT();
writer.SaveChunk(parents);

// Write
the
SSTR
chunk.
if (HasSharedStrings)
writer.SaveChunk(SSTR, 0);

// Write
the
META
chunk.
if (HasMetadata)
writer.SaveChunk(META, 0);

// Write
the
SIGN
chunk.
if (HasSignatures)
writer.SaveChunk(SIGN);

// Write
the
END
chunk. \
    writer.WriteChunk("END", "</roblox>");
}

// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //
// Write
the
chunk
buffers
with the header data
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //

using (BinaryWriter writer = new BinaryWriter(stream))
{
stream.Position = 0;
stream.SetLength(0);

writer.Write(MagicHeader
.Select(ch = > (byte)ch)
.ToArray());

writer.Write(Version);
writer.Write(NumClasses);
writer.Write(NumInstances);
writer.Write(Reserved);

foreach (BinaryRobloxFileChunk chunk in Chunks)
{
if (chunk.HasWriteBuffer)
{
byte[] writeBuffer = chunk.WriteBuffer;
writer.Write(writeBuffer);
}
}
}
}
}
}
"""










