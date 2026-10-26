"""Local, offline previews. No masks invented for unannotated frames."""
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = {1: (255, 110, 80), 2: (80, 220, 100), 3: (70, 160, 255)}


def gray(a, limits=None):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [1, 99]) if limits is None else limits
    b = np.clip((a-lo) / max(float(hi-lo), 1e-12) * 255, 0, 255).astype('uint8')
    return Image.fromarray(b).convert('RGB')


def overlay(image, mask):
    arr = np.array(image).astype(float)
    for label, color in COLORS.items():
        where = mask == label
        arr[where] = .6 * arr[where] + .4 * np.array(color)
    return Image.fromarray(arr.astype('uint8'))


def tile(image, text, width=240, height=180, stretch=False):
    panel = Image.new('RGB', (width, height+28), 'black')
    if stretch:
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    else:
        factor = min(width/image.width, height/image.height)
        image = image.resize((max(1, round(image.width*factor)), max(1, round(image.height*factor))), Image.Resampling.BILINEAR)
    panel.paste(image, ((width-image.width)//2, 28+(height-image.height)//2))
    ImageDraw.Draw(panel).text((5, 5), text, fill='white')
    return panel


def raw_previews(folder, cine, masks, phase_frames, slices):
    folder = Path(folder)
    zlist = sorted(set([0, cine.shape[1]//2, cine.shape[1]-1] +
                       [r['slice_z'] for r in slices if r['flags']][:2]))
    frames = []
    limits = {z: np.percentile(cine[:, z], [1, 99]) for z in zlist}
    for t in range(len(cine)):
        canvas = Image.new('RGB', (480, 208*len(zlist)))
        for j, z in enumerate(zlist):
            native = gray(cine[t, z], (cine[:, z].min(), cine[:, z].max()))
            im = gray(cine[t, z], limits[z])
            phase = next((k for k, v in phase_frames.items() if v == t and k in masks), None)
            source = overlay(native, masks[phase][z]) if phase else native
            canvas.paste(tile(source, f'Native t={t} z={z} '+(phase or 'NO MASK')), (0, j*208))
            # Exact aspect ratio of model input shown on right. Display downsample only.
            canvas.paste(tile(im, 'Model 512x384 aspect; p01/p99', stretch=True), (240, j*208))
        frames.append(canvas)
    frames[0].save(folder / 'cine.gif', save_all=True, append_images=frames[1:], duration=160, loop=0)
    for phase, mask in masks.items():
        sheets = []
        for z in range(cine.shape[1]):
            image = gray(cine[phase_frames[phase], z], np.percentile(cine[:, z], [1, 99]))
            sheets.append(tile(overlay(image, mask[z]), f'{phase} frame={phase_frames[phase]} z={z}'))
        page = Image.new('RGB', (240*4, 208*((len(sheets)+3)//4)))
        for i, sheet in enumerate(sheets): page.paste(sheet, ((i % 4)*240, (i//4)*208))
        page.save(folder / (phase + '.png'))
    return zlist


def trajectory_preview(path, video, tracks, visibility):
    # All queries are drawn, no cherry-picked successful tracks. White = low visibility.
    frames = []
    p = np.asarray(tracks)[0]
    v = np.asarray(visibility)[0]
    for t in range(len(video)):
        a = np.clip(video[t, 0], 0, 255).astype('uint8')
        im = Image.fromarray(a).convert('RGB')
        draw = ImageDraw.Draw(im)
        for n in range(p.shape[1]):
            color = (60, 255, 120) if v[t, n] > .9 else (240, 240, 240)
            x, y = p[t, n]
            if 0 <= x < im.width and 0 <= y < im.height:
                draw.ellipse((x-2, y-2, x+2, y+2), outline=color)
            for k in range(max(1, t-3), t+1):
                pts = p[k-1:k+1, n]
                if ((pts >= 0) & (pts < [im.width, im.height])).all():
                    draw.line([tuple(pts[0]), tuple(pts[1])], fill=color)
        draw.rectangle((0, 0, 510, 18), fill='black')
        draw.text((3, 3), f'Clip frame {t}; green V>.9, white otherwise; trails <=4 frames', fill='white')
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=160, loop=0)


def document(title, body):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(title)+
            '</title><style>body{font:16px/1.6 sans-serif;max-width:1100px;margin:32px auto;padding:0 20px}'
            'table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:6px}'
            'img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
            'a{color:#176ca4}.warn{color:#a43a16}</style><h1>'+html.escape(title)+'</h1>'+body+'</html>')


def table(rows, fields):
    out = '<table><tr>'+''.join('<th>'+html.escape(f)+'</th>' for f in fields)+'</tr>'
    for r in rows:
        out += '<tr>'+''.join('<td>'+html.escape(str(r.get(f, '')))+'</td>' for f in fields)+'</tr>'
    return out+'</table>'
