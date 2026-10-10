"""sarflood_utils.py
ฟังก์ชันกลางของชุดการสอน Teach_SAR_flood_Mapping_ML (notebook 01-05)

แบ่งเป็น 4 กลุ่ม
1. อ่านข้อมูล            load_s1, read_on_grid, load_boundaries
2. ประมวลผล SAR         db2lin, lin2db, speckle_filter, clean_mask, flood_change_map
3. Machine learning     make_features, sample_pseudo_labels
4. แผนที่มาตรฐานวารสาร   journal_map, add_inset, save_fig
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy import ndimage
from pyproj import Transformer
import matplotlib.pyplot as plt

FLOOD_TAG = '20171017'      # ภาพน้ำท่วม
REF_TAG = '20180109'        # ภาพอ้างอิง
RAI_M2 = 1600.0             # 1 ไร่ = 1,600 ตร.ม.
DIFF_THRESHOLD = -3.0       # dB ค่าเริ่มต้นของ threshold (ปรับได้หลังดู histogram ใน notebook 02)

# รหัส ESA WorldCover: (ชื่อ, สี)
WORLDCOVER = {
    10: ('Tree cover', '#006400'), 20: ('Shrubland', '#ffbb22'), 30: ('Grassland', '#ffff4c'),
    40: ('Cropland', '#f096ff'), 50: ('Built-up', '#fa0000'), 60: ('Bare / sparse', '#b4b4b4'),
    80: ('Permanent water', '#0064c8'), 90: ('Herbaceous wetland', '#0096a0'),
    95: ('Mangroves', '#00cf75'), 100: ('Moss and lichen', '#fae6a0'),
}

FEATURE_SETS = {
    'sar_4band': ['VV_flood', 'VH_flood', 'VV_ref', 'VH_ref'],
    'sar_with_ratio': ['VV_flood', 'VH_flood', 'VV_ref', 'VH_ref', 'CR_flood', 'CR_ref'],
    'sar_ratio_terrain': ['VV_flood', 'VH_flood', 'VV_ref', 'VH_ref', 'CR_flood', 'CR_ref',
                          'elevation', 'slope'],
}


# ---------------------------------------------------------------- 1. อ่านข้อมูล
_AOI_CACHE = {}


def aoi_mask(data_dir, grid):
    """หน้ากากพื้นที่ศึกษา (True = อยู่ในขอบเขตอำเภอ) บนกริดที่กำหนด
    จำเป็นเพราะ GeoTIFF ที่ส่งออกจาก GEE เก็บจุดภาพนอกพื้นที่ศึกษาเป็นค่า 0 ไม่ใช่ค่า nodata"""
    key = (os.path.abspath(data_dir), tuple(grid['transform'])[:6], grid['height'], grid['width'])
    if key not in _AOI_CACHE:
        gdf = gpd.read_file(os.path.join(data_dir, 'boundary_amphoe.geojson')).to_crs(grid['crs'])
        _AOI_CACHE[key] = rasterize([(g, 1) for g in gdf.geometry],
                                    out_shape=(grid['height'], grid['width']),
                                    transform=grid['transform'], fill=0, dtype='uint8') == 1
    return _AOI_CACHE[key]


def load_s1(data_dir, tag, pol):
    """อ่านภาพ Sentinel-1 หนึ่งแบนด์ คืนค่า (อาร์เรย์ dB แบบ float32, grid)
    ไฟล์เก็บค่า dB x 100 แบบ int16 จึงหารด้วย 100 กลับเป็น dB จุดภาพนอกพื้นที่ศึกษาเป็น NaN"""
    with rasterio.open(os.path.join(data_dir, f'S1_{tag}_{pol}.tif')) as src:
        arr = src.read(1, masked=True)
        grid = {'transform': src.transform, 'crs': src.crs, 'height': src.height, 'width': src.width}
    arr = (arr.astype('float32') / 100.0).filled(np.nan)
    arr[~aoi_mask(data_dir, grid)] = np.nan
    return arr, grid


def read_on_grid(path, grid, resampling='nearest', scale=1.0):
    """อ่านราสเตอร์ใด ๆ แล้วปรับให้อยู่บนกริดเดียวกับภาพ Sentinel-1
    resampling: 'nearest' สำหรับข้อมูลประเภท (LULC) 'bilinear' สำหรับข้อมูลต่อเนื่อง (DEM)"""
    dst = np.full((grid['height'], grid['width']), np.nan, dtype='float32')
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_nodata=src.nodata, dst_nodata=np.nan,
                  dst_transform=grid['transform'], dst_crs=grid['crs'],
                  resampling=getattr(Resampling, resampling))
    dst[~aoi_mask(os.path.dirname(path), grid)] = np.nan
    return dst * scale


def load_boundaries(data_dir, crs):
    """อ่านขอบเขตอำเภอและจังหวัด แล้วแปลงเป็นระบบพิกัดเดียวกับราสเตอร์"""
    amphoe = gpd.read_file(os.path.join(data_dir, 'boundary_amphoe.geojson')).to_crs(crs)
    province = gpd.read_file(os.path.join(data_dir, 'boundary_province.geojson')).to_crs(crs)
    return amphoe, province


def pixel_area_m2(grid):
    """พื้นที่ของหนึ่งจุดภาพ (ตร.ม.)"""
    t = grid['transform']
    return abs(t.a * t.e)


def extent(grid):
    """ขอบเขตภาพสำหรับ imshow: [ซ้าย, ขวา, ล่าง, บน]"""
    t = grid['transform']
    return [t.c, t.c + t.a * grid['width'], t.f + t.e * grid['height'], t.f]


def rasterize_zones(gdf, grid):
    """แปลงโพลิกอนเป็นราสเตอร์รหัสโซน (1, 2, ... ตามลำดับแถวใน gdf; 0 = นอกโซน)"""
    shapes = [(geom, i + 1) for i, geom in enumerate(gdf.geometry)]
    return rasterize(shapes, out_shape=(grid['height'], grid['width']),
                     transform=grid['transform'], fill=0, dtype='uint8')


def write_geotiff(path, arr, grid, nodata=255):
    """บันทึกอาร์เรย์ uint8 เป็น GeoTIFF แบบบีบอัด"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', height=grid['height'], width=grid['width'],
                       count=1, dtype='uint8', crs=grid['crs'], transform=grid['transform'],
                       nodata=nodata, compress='deflate') as dst:
        dst.write(arr.astype('uint8'), 1)


# ---------------------------------------------------------------- 2. ประมวลผล SAR
def db2lin(db):
    """dB -> linear power"""
    return 10.0 ** (db / 10.0)


def lin2db(lin):
    """linear power -> dB"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(lin)


def speckle_filter(db, size=7):
    """กรอง speckle ด้วยค่าเฉลี่ยเคลื่อนที่ (boxcar) ขนาด size x size จุดภาพ
    เฉลี่ยบนค่า linear power แล้วแปลงกลับเป็น dB และไม่นำ NaN มาคิด"""
    valid = np.isfinite(db)
    lin = np.where(valid, db2lin(db), 0.0).astype('float32')
    num = ndimage.uniform_filter(lin, size=size, mode='constant')
    den = ndimage.uniform_filter(valid.astype('float32'), size=size, mode='constant')
    with np.errstate(divide='ignore', invalid='ignore'):
        out = lin2db(num / den)
    out[~valid] = np.nan
    return out.astype('float32')


def clean_mask(mask, min_pixels=8):
    """ลบกลุ่มจุดภาพที่ต่อเนื่องกัน (8 ทิศ) น้อยกว่า min_pixels จุดภาพ"""
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labels]


def flood_change_map(data_dir, threshold=DIFF_THRESHOLD, pol='VH', filter_size=7,
                     perm_months=10, max_slope=5.0, min_pixels=8):
    """แผนที่น้ำท่วมแบบ change detection ตามขั้นตอนใน notebook 02
    คืนค่า dict: ref, flood_img (dB หลังกรอง), diff (dB), raw (ก่อนปรับแต่ง), flood (ผลสุดท้าย),
    valid, perm (น้ำถาวร), slope (องศา), grid"""
    ref_raw, grid = load_s1(data_dir, REF_TAG, pol)
    fld_raw, _ = load_s1(data_dir, FLOOD_TAG, pol)
    ref = speckle_filter(ref_raw, filter_size)
    fld = speckle_filter(fld_raw, filter_size)
    diff = fld - ref
    valid = np.isfinite(diff)

    season = read_on_grid(os.path.join(data_dir, 'jrc_seasonality.tif'), grid, 'nearest')
    slope = read_on_grid(os.path.join(data_dir, 'slope_srtm.tif'), grid, 'bilinear', scale=0.01)
    perm = np.nan_to_num(season, nan=0) >= perm_months

    raw = valid & (diff < threshold)
    flood = raw & ~perm & (np.nan_to_num(slope, nan=0) < max_slope)
    flood = clean_mask(flood, min_pixels)
    return {'ref': ref, 'flood_img': fld, 'diff': diff, 'raw': raw, 'flood': flood,
            'valid': valid, 'perm': perm, 'slope': slope, 'grid': grid, 'threshold': threshold}


# ---------------------------------------------------------------- 3. Machine learning
def make_features(data_dir):
    """สร้างชั้นข้อมูล feature ทั้งหมดบนกริดเดียวกัน คืนค่า (dict ชื่อ -> อาร์เรย์ 2 มิติ, grid)
    ใช้แบนด์ดิบ (ไม่กรอง speckle) และ CR = VH - VV ในหน่วย dB (cross-polarization ratio)"""
    f = {}
    f['VV_flood'], grid = load_s1(data_dir, FLOOD_TAG, 'VV')
    f['VH_flood'], _ = load_s1(data_dir, FLOOD_TAG, 'VH')
    f['VV_ref'], _ = load_s1(data_dir, REF_TAG, 'VV')
    f['VH_ref'], _ = load_s1(data_dir, REF_TAG, 'VH')
    f['CR_flood'] = f['VH_flood'] - f['VV_flood']
    f['CR_ref'] = f['VH_ref'] - f['VV_ref']
    f['elevation'] = read_on_grid(os.path.join(data_dir, 'dem_srtm.tif'), grid, 'bilinear')
    f['slope'] = read_on_grid(os.path.join(data_dir, 'slope_srtm.tif'), grid, 'bilinear', scale=0.01)
    return f, grid


def sample_pseudo_labels(fm, feats, n_per_class=10000, margin=1.5, block_km=5.0, seed=42):
    """สุ่มจุดตัวอย่างพร้อม pseudo-label จากแผนที่ threshold
    - ท่วม (1):    อยู่ในแผนที่น้ำท่วม และ diff < threshold - margin
    - ไม่ท่วม (0): diff > threshold + margin และไม่ใช่น้ำถาวร
    - จุดภาพในช่วง threshold +/- margin ถือว่ากำกวม ไม่นำมาใช้
    คืนค่า DataFrame: row, col, x, y, block, label และคอลัมน์ feature"""
    grid, t = fm['grid'], fm['threshold']
    ok = fm['valid'].copy()
    for a in feats.values():
        ok &= np.isfinite(a)
    cand = {1: ok & fm['flood'] & (fm['diff'] < t - margin),
            0: ok & ~fm['perm'] & ~fm['raw'] & (fm['diff'] > t + margin)}

    rng = np.random.default_rng(seed)
    rows, cols, labels = [], [], []
    for label, m in cand.items():
        idx = np.flatnonzero(m)
        pick = rng.choice(idx, size=min(n_per_class, idx.size), replace=False)
        r, c = np.unravel_index(pick, m.shape)
        rows.append(r); cols.append(c); labels.append(np.full(r.size, label))
    rows, cols, labels = map(np.concatenate, (rows, cols, labels))

    tr = grid['transform']
    nb = max(1, int(round(block_km * 1000 / abs(tr.a))))      # จำนวนจุดภาพต่อด้านของบล็อก
    df = pd.DataFrame({'row': rows, 'col': cols,
                       'x': tr.c + (cols + 0.5) * tr.a, 'y': tr.f + (rows + 0.5) * tr.e,
                       'block': (rows // nb) * 1000 + (cols // nb), 'label': labels})
    for name, a in feats.items():
        df[name] = a[rows, cols]
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def predict_raster(model, feats, names, valid, chunk=2_000_000):
    """ทำนายความน่าจะเป็นของน้ำท่วมทั้งภาพ โดยแบ่งทำทีละส่วนเพื่อประหยัดหน่วยความจำ"""
    ok = valid.copy()
    for n in names:
        ok &= np.isfinite(feats[n])
    idx = np.flatnonzero(ok)
    prob = np.full(valid.shape, np.nan, dtype='float32')
    for s in range(0, idx.size, chunk):
        part = idx[s:s + chunk]
        X = np.column_stack([feats[n].ravel()[part] for n in names])
        prob.ravel()[part] = model.predict_proba(pd.DataFrame(X, columns=names))[:, 1]
    return prob


# ---------------------------------------------------------------- 4. แผนที่มาตรฐานวารสาร
def journal_map(ax, grid, amphoe, province=None, lonlat_step=0.1, scalebar_km=10,
                north=True, label_amphoe=True, edge='black', pad_left=0.0):
    """เติมองค์ประกอบแผนที่ลงบนแกน ax ที่พล็อตราสเตอร์ไว้แล้ว:
    ขอบเขตจังหวัด/อำเภอ, กริดพิกัดภูมิศาสตร์, มาตราส่วน, ทิศเหนือ, ชื่ออำเภอ
    pad_left: ขยายกรอบแผนที่ไปทางซ้าย (สัดส่วนของความกว้าง) เพื่อเว้นที่ให้แผนที่แทรกและคำอธิบายสัญลักษณ์"""
    left, right, bottom, top = extent(grid)
    left -= pad_left * (right - left)
    if province is not None:
        province.boundary.plot(ax=ax, color='dimgray', linewidth=1.6, linestyle='--', zorder=5)
    amphoe.boundary.plot(ax=ax, color=edge, linewidth=1.0, zorder=6)
    ax.set_xlim(left, right); ax.set_ylim(bottom, top)

    # กริดพิกัด: คำนวณตำแหน่งเส้นลองจิจูด/ละติจูดบนระบบ UTM
    to_ll = Transformer.from_crs(grid['crs'], 'EPSG:4326', always_xy=True)
    to_xy = Transformer.from_crs('EPSG:4326', grid['crs'], always_xy=True)
    lon0, lat0 = to_ll.transform(left, bottom)
    lon1, lat1 = to_ll.transform(right, top)
    lons = np.arange(np.ceil(lon0 / lonlat_step) * lonlat_step, lon1, lonlat_step)
    lats = np.arange(np.ceil(lat0 / lonlat_step) * lonlat_step, lat1, lonlat_step)
    ax.set_xticks([to_xy.transform(lo, lat0)[0] for lo in lons])
    nd = 1 if lonlat_step >= 0.1 else 2                # จำนวนทศนิยมของป้ายพิกัด
    ax.set_xticklabels([f'{lo:.{nd}f}°E' for lo in lons])
    ax.set_yticks([to_xy.transform(lon0, la)[1] for la in lats])
    ax.set_yticklabels([f'{la:.{nd}f}°N' for la in lats], rotation=90, va='center')
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)
    ax.grid(color='gray', linewidth=0.3, linestyle=':')

    if label_amphoe and 'ADM2_NAME' in amphoe:
        for _, r in amphoe.iterrows():
            p = r.geometry.representative_point()
            ax.text(p.x, p.y, r['ADM2_NAME'], fontsize=8, ha='center', weight='bold', zorder=7,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

    w, h = right - left, top - bottom
    if scalebar_km:                                   # มาตราส่วนแบบแท่ง มุมล่างขวา
        while scalebar_km * 1000 > 0.3 * w and scalebar_km > 1:   # ย่อให้พอดีกับความกว้างแผนที่
            scalebar_km = {10: 5, 5: 2, 2: 1}.get(scalebar_km, max(1, scalebar_km // 2))
        length = scalebar_km * 1000
        x0, y0 = right - 0.05 * w - length, bottom + 0.05 * h
        ax.add_patch(plt.Rectangle((x0 - 0.02 * w, y0 - 0.02 * h), length + 0.04 * w, 0.08 * h,
                                   facecolor='white', alpha=0.8, edgecolor='none', zorder=8))
        ax.plot([x0, x0 + length], [y0, y0], color='black', linewidth=3, solid_capstyle='butt', zorder=9)
        ax.text(x0 + length / 2, y0 + 0.012 * h, f'{scalebar_km} km', ha='center', va='bottom',
                fontsize=8, zorder=9)
    if north:                                         # ลูกศรทิศเหนือ มุมบนขวา
        ax.annotate('N', xy=(0.95, 0.96), xytext=(0.95, 0.86), xycoords='axes fraction',
                    ha='center', va='center', fontsize=11, weight='bold', zorder=9,
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=1.5))
    ax.set_aspect('equal'); ax.set_xlabel(''); ax.set_ylabel('')
    return ax


def add_inset(ax, amphoe, province, loc=(0.02, 0.02, 0.26, 0.26)):
    """แผนที่แทรกแสดงตำแหน่งพื้นที่ศึกษาภายในจังหวัด"""
    ins = ax.inset_axes(loc)
    province.plot(ax=ins, facecolor='whitesmoke', edgecolor='dimgray', linewidth=0.6)
    amphoe.plot(ax=ins, facecolor='crimson', edgecolor='black', linewidth=0.4)
    for _, r in province.iterrows():
        p = r.geometry.representative_point()
        ins.text(p.x, p.y, r.get('ADM1_NAME', ''), fontsize=5.5, ha='center')
    ins.set_xticks([]); ins.set_yticks([]); ins.set_aspect('equal')
    ins.set_xlabel(''); ins.set_ylabel('')
    ins.set_facecolor('white'); ins.set_zorder(10)
    return ins


def save_fig(fig, name, out_dir='outputs', dpi=300):
    """บันทึกรูปที่ 300 dpi ตามมาตรฐานวารสาร"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print('บันทึก:', path)
    return path


def show_mask(ax, mask, grid, color, alpha=1.0):
    """ซ้อนหน้ากาก (True/False) ด้วยสีเดียวลงบนแผนที่"""
    from matplotlib.colors import ListedColormap
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]), alpha=alpha,
              extent=extent(grid), interpolation='nearest')


def stretch(a, lo, hi):
    """ยืดค่าช่วง [lo, hi] ให้เป็น [0, 1] สำหรับภาพผสมสี (NaN กลายเป็นสีขาว)"""
    out = np.clip((a - lo) / (hi - lo), 0, 1)
    return np.where(np.isfinite(a), out, 1.0)


def spatial_split(df, test_size=0.3, seed=42):
    """แบ่ง train/test เชิงพื้นที่: จุดตัวอย่างในบล็อกเดียวกันอยู่ฝั่งเดียวกันเสมอ"""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, df['label'], df['block']))
    return df.iloc[tr].copy(), df.iloc[te].copy()
