import argparse
import os
import struct
import zlib
import numpy as np


def write_png(path, data):
    h, w = data.shape[:2]
    def chunk(ctype, cdata):
        c = ctype + cdata
        return struct.pack('>I', len(cdata)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    raw = b''.join(b'\x00' + data[y].tobytes() for y in range(h))
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b'IDAT', zlib.compress(raw, 1)))
        f.write(chunk(b'IEND', b''))


def read_png(path):
    import struct, zlib
    with open(path, 'rb') as f:
        sig = f.read(8)
        assert sig == b'\x89PNG\r\n\x1a\n'
        chunks = {}
        while True:
            length = struct.unpack('>I', f.read(4))[0]
            ctype = f.read(4)
            cdata = f.read(length)
            f.read(4)
            chunks.setdefault(ctype, b'')
            chunks[ctype] += cdata
            if ctype == b'IEND':
                break
        w, h, _, _ = struct.unpack('>IIBB', chunks[b'IHDR'][:10])
        raw = zlib.decompress(chunks[b'IDAT'])
        stride = w * 3 + 1
        rows = [raw[y * stride + 1: y * stride + stride] for y in range(h)]
        img = np.frombuffer(b''.join(rows), dtype=np.uint8).reshape(h, w, 3)
    return img


def _build_nipy_spectral(n=1556):
    t = np.linspace(0, 1, n)
    r = np.zeros(n); g = np.zeros(n); b = np.zeros(n)
    m1 = t < 0.1;  r[m1] = 0.4+t[m1]*3;  b[m1] = 0.4+t[m1]*3
    m2 = (t>=0.1)&(t<0.2); r[m2]=0.7; b[m2]=0.7-t[m2]*3
    m3 = (t>=0.2)&(t<0.4); r[m3]=0.7-(t[m3]-0.2)*3.5; g[m3]=(t[m3]-0.2)*5; b[m3]=0.1
    m4 = (t>=0.4)&(t<0.6); g[m4]=1.0; b[m4]=(t[m4]-0.4)*5
    m5 = (t>=0.6)&(t<0.75); r[m5]=(t[m5]-0.6)*7; g[m5]=1.0-(t[m5]-0.6)*3; b[m5]=1.0
    m6 = (t>=0.75)&(t<0.9); r[m6]=1.0; g[m6]=0.55-(t[m6]-0.75)*3; b[m6]=1.0-(t[m6]-0.75)*7
    m7 = t>=0.9; r[m7]=1.0; g[m7]=0.1+(t[m7]-0.9)*2
    return np.clip(np.stack([r, g, b], axis=1), 0, 1)


def resize_nearest(img, size):
    h, w = img.shape[:2]
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return img[np.ix_(ys, xs)]


def colorize_vertex_map(p2v, valid_mask):
    cmap = _build_nipy_spectral(1556)
    h, w = valid_mask.shape
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    idx = np.clip(p2v[:, :, 0], 0, 1555)
    colored = (cmap[idx] * 255).astype(np.uint8)
    img[valid_mask] = colored[valid_mask]
    return img


def make_contact_overlay(p2v, valid_mask, contact_verts):
    h, w = valid_mask.shape
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    contact_px = valid_mask & np.isin(p2v[:, :, 0], contact_verts)
    img[valid_mask & ~contact_px] = [55, 55, 75]
    img[contact_px] = [220, 50, 50]
    return img


def _thermal_cmap(t):
    # t in [0,1]: 0=black (far/cold), 0.5=red, 0.75=orange, 1=yellow (close/hot)
    r = np.clip(t * 2.5, 0, 1)
    g = np.clip(t * 2.5 - 1.0, 0, 1)
    b = np.clip(t * 2.5 - 2.0, 0, 1)
    return np.stack([r, g, b], axis=1)


def make_distance_heatmap(p2v, bary, valid_mask, vertex_distances):
    n = len(vertex_distances)
    d = vertex_distances.copy().astype(np.float32)

    h, w = valid_mask.shape
    img = np.full((h, w, 3), 20, dtype=np.uint8)

    v0 = np.clip(p2v[:, :, 0], 0, n - 1)
    v1 = np.clip(p2v[:, :, 1], 0, n - 1)
    v2 = np.clip(p2v[:, :, 2], 0, n - 1)
    dist_px = bary[:, :, 0] * d[v0] + bary[:, :, 1] * d[v1] + bary[:, :, 2] * d[v2]

    # closeness = 1 - distance; far vertices stay black, close ones glow yellow→red
    closeness = 1.0 - dist_px
    cmap = _thermal_cmap(np.linspace(0, 1, 256))
    cmap_idx = np.clip((closeness * 255).astype(int), 0, 255)
    colored = (cmap[cmap_idx] * 255).astype(np.uint8)
    img[valid_mask] = colored[valid_mask]
    return img


def make_synthetic_distances(n_verts=1556, seed=42):
    rng = np.random.default_rng(seed)
    distances = np.ones(n_verts, dtype=np.float32)

    # right hand only (vertices 0-777): gradient fingertips=close, wrist=far
    t = np.linspace(0, 1, 778)
    d_right = 0.15 + 0.75 * t  # base gradient: close at 0, far at 777
    # 5 fingertip dips to simulate contact at individual fingertips
    for c in [0.04, 0.11, 0.18, 0.25, 0.32]:
        d_right -= 0.12 * np.exp(-((t - c) ** 2) / 0.0008)
    d_right += 0.02 * rng.standard_normal(778).astype(np.float32)
    distances[:778] = np.clip(d_right, 0, 1)
    # left hand stays at 1.0 (far = black in thermal map)
    return distances


def add_border(img, color=(80, 80, 100), thickness=3):
    out = img.copy()
    out[:thickness, :] = color
    out[-thickness:, :] = color
    out[:, :thickness] = color
    out[:, -thickness:] = color
    return out


def hstack_with_arrow(panels, arrow_w=50, bg=20):
    h = panels[0].shape[0]
    total_w = sum(p.shape[1] for p in panels) + arrow_w * (len(panels) - 1)
    out = np.full((h, total_w, 3), bg, dtype=np.uint8)
    x = 0
    for i, p in enumerate(panels):
        out[:, x:x + p.shape[1]] = p
        x += p.shape[1]
        if i < len(panels) - 1:
            mid = h // 2
            for dy in range(-2, 3):
                out[mid + dy, x + 5:x + arrow_w - 10] = [200, 200, 200]
            for k in range(12):
                out[mid - k:mid + k + 1, x + arrow_w - 10 - k] = [200, 200, 200]
            x += arrow_w
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/hcontact_mano_rest")
    parser.add_argument("--view", default="palm", choices=["palm", "back", "left", "right"])
    parser.add_argument("--out", default="data/hcontact_mano_rest/lookup_table_figure.png")
    parser.add_argument("--mode", default="lookup", choices=["lookup", "heatmap"],
                        help="lookup: show the 3-panel lookup table figure. heatmap: show distance heatmap for Goal 2.")
    args = parser.parse_args()

    p2v = np.load(os.path.join(args.data_dir, "pixel_to_vertex_map_1024.npz"))[args.view]
    bary = np.load(os.path.join(args.data_dir, "bary_coords_map_1024.npz"))[args.view]
    valid_mask = p2v[:, :, 0] >= 0
    render = read_png(os.path.join(args.data_dir, f"body_render_norm_{args.view}.png"))

    size = 512

    if args.mode == "heatmap":
        distances = make_synthetic_distances()
        heatmap_img = make_distance_heatmap(p2v, bary, valid_mask, distances)
        panels = [
            add_border(resize_nearest(render, size)),
            add_border(resize_nearest(heatmap_img, size)),
        ]
        result = hstack_with_arrow(panels)
        out = args.out.replace("lookup_table_figure", "heatmap_figure")
        write_png(out, result)
        print(f"Saved -> {out}")
        write_png(os.path.join(args.data_dir, "panel_heatmap.png"), resize_nearest(heatmap_img, size))
        print(f"Saved -> {os.path.join(args.data_dir, 'panel_heatmap.png')}")
    else:
        vertex_map = colorize_vertex_map(p2v, valid_mask)
        rng = np.random.default_rng(42)
        contact_verts = rng.choice(np.arange(0, 180), size=55, replace=False)
        contact_img = make_contact_overlay(p2v, valid_mask, contact_verts)
        panels = [add_border(resize_nearest(img, size)) for img in [render, vertex_map, contact_img]]
        result = hstack_with_arrow(panels)
        write_png(args.out, result)
        print(f"Saved -> {args.out}")
        for name, img in [("render", render), ("vertex_map", vertex_map), ("contact_lift", contact_img)]:
            out_path = os.path.join(args.data_dir, f"panel_{name}.png")
            write_png(out_path, resize_nearest(img, size))
            print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
