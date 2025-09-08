from pathlib import Path
import numpy as np

def ReadImage(f):
    if isinstance(f, str) or isinstance(f, Path):
        with open(f, "rb") as f:
            return ReadImage(f)
    header = {}
    data = []
    for line in f:
        line = line.decode("utf-8").strip()
        if " = " in line:
            key, value = line.split(" = ", 1)
            header[key] = value
        if "ElementDataFile" in line:
            if "LOCAL" in line:
                data = f.read()
            else: # follow pointer to raw file
                raw_file = header["ElementDataFile"]
                with open(raw_file, "rb") as raw_f:
                    data = raw_f.read()
            break

    
    return {
        "header": header,
        "data": data
    }

def GetArrayFromImage(image):
    header = image["header"]
    data = image["data"]
    
    # Check if data is compressed and decompress if necessary
    if "CompressedData" in header and header["CompressedData"] == "True":
        import zlib
        data = zlib.decompress(data)

    dtype = np.uint8
    if "ElementType" in header and header["ElementType"] == "MET_FLOAT":
        dtype = np.float32
    elif "ElementType" in header and header["ElementType"] == "MET_DOUBLE":
        dtype = np.float64
    elif "ElementType" in header and header["ElementType"] == "MET_INT":
        dtype = np.int32
    elif "ElementType" in header and header["ElementType"] == "MET_UINT":
        dtype = np.uint32
    elif "ElementType" in header and header["ElementType"] == "MET_SHORT":
        dtype = np.int16
    elif "ElementType" in header and header["ElementType"] == "MET_USHORT":
        dtype = np.uint16
    elif "ElementType" in header and header["ElementType"] == "MET_CHAR":
        dtype = np.int8
    elif "ElementType" in header and header["ElementType"] == "MET_UCHAR":
        dtype = np.uint8
     
    data = np.frombuffer(data, dtype=dtype)

    shape = [int(i) for i in header["DimSize"].split(" ")][::-1]

    data = data.reshape(shape)

    return data