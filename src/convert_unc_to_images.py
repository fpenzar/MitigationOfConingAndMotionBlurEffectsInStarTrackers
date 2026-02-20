import struct
import numpy as np
import matplotlib.pyplot as plt
from data_io import DataIO
 
def read_unc_images(path):
    dataIo = DataIO()
    with open(path, "rb") as f:
        f.read(8)
        while True:
            header = f.read(34)
            if len(header) < 34:
                break

            endian = "<"
            integration_time = struct.unpack_from(f"{endian}B", header, 4)[0]
            valid_tag = struct.unpack_from(f"{endian}H", header, 10)[0]
            valid = valid_tag == 21840
            sub_timestamp = struct.unpack_from(f"{endian}H", header, 22)[0] / 65536.0
            timestamp = struct.unpack_from(f"{endian}I", header, 24)[0]
            H = struct.unpack_from(f"{endian}H", header, 28)[0]
            W = struct.unpack_from(f"{endian}H", header, 30)[0]
            IMOD = struct.unpack_from(f"{endian}H", header, 32)[0]

            H = (IMOD + 1) * H
            frame_bytes = H * W

            data = f.read(frame_bytes)
            if len(data) < frame_bytes:
                break

            img = np.frombuffer(data, dtype=np.uint8)

            img = img.reshape(H, W)

            img = np.fliplr(img)

            full_ts = timestamp + sub_timestamp

            # dataIo.save_image(img, full_ts)

            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            plt.title(f"Timestamp: {full_ts:.6f}")
            plt.pause(0.1)
            plt.clf()

            yield img, full_ts, valid


PATH = r"..."
for img, ts, valid in read_unc_images(PATH):
    print("Frame", ts, img.shape, valid)
