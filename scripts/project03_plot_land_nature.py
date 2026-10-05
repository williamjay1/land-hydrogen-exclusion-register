"""Nature-inspired, evidence-preserving Land artwork; no new model estimates.

Uses a pinned, reviewed SciencePlots nature style, with explicit overrides for
this manuscript. Exports genuine vector PDF/SVG and native 1200 ppi RGB PNG.
The geometry audit is a layout screen, not an accessibility certification.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image
import fitz
from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import transform

ROOT = Path(r"D:\MLWork\Europe_green_hydrogen_siting")
RAW = Path(r"F:\AcademicData\Project03\raw\nature_visual_resources_20261005_v1")
RESULT = ROOT / "results/land_shortlist_assessment_20261005_v2"
OUT = ROOT / "final_submission_seta/figures_land_nature_20261005_v1"
QA = ROOT / "results/land_nature_artwork_20261005_v1"
PREVIEW = ROOT / "temp/land_nature_artwork_20261005"
LAND = ROOT / "datasets/intermediate/land_context_clc2018_natura2024_20261005_v1/land_context_metrics.csv"
ENERGY = ROOT / "results/regret_certificate_pilot_20261002_v1/run_20261002T063739Z_u50_fourweek/annual_candidate_costs_and_bounds.csv"
GIS = Path(r"E:\AcademicData\Project03\raw\gisco\NUTS_RG_01M_2024_4326_LEVL_3.geojson")
SOURCE = ROOT / "final_submission_seta/bound_certified_hydrogen_screening_pilot_20261002_v1.tex"
ZONES = ["BE", "DE_LU", "FR", "NL"]
NAMES = {"BE": "Belgium", "DE_LU": "Germany-Luxembourg", "FR": "Mainland France", "NL": "Netherlands"}
VIEWS = ["industrial_context", "agricultural_context", "natural_context", "protected_context"]
CONTEXTS = ["industrial_commercial_fraction", "agricultural_fraction", "forest_seminatural_fraction", "natura2000_union_fraction"]
CONTEXT_LABELS = ["Industrial /\ncommercial", "Agricultural", "Forest /\nsemi-natural", "Natura 2000"]
COST = "forecast_objective_eur_per_kg_2023"
SCENARIOS = ["u50_fourweek", "u30_fourweek", "u70_fourweek", "u50_lastweek"]
SCENARIO_LABELS = ["50% / four-week", "30% / four-week", "70% / four-week", "50% / prior-week"]
SCALES = ["nuts3", "buffer_20km_clipped", "buffer_10km_clipped", "buffer_5km_clipped"]
WIDTH_MM, DPI = 165, 1200
BLUE, ORANGE, GREY, INK = "#0072B2", "#D55E00", "#9D9D9D", "#202020"
RECORDS, GEOMETRY_REPORTS = [], []


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def configure():
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.style.use(str(RAW / "scienceplots_nature.mplstyle"))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "axes.titleweight": "normal", "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .55, "axes.edgecolor": INK, "axes.grid": False,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": .55, "ytick.major.width": .55,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.usetex": False, "mathtext.fontset": "dejavusans",
        "savefig.bbox": None, "savefig.facecolor": "white",
        "savefig.transparent": False, "figure.facecolor": "white",
        "figure.dpi": 150, "path.simplify": False,
    })


def figure(height):
    return plt.figure(figsize=(WIDTH_MM / 25.4, height / 25.4))


def title(fig, ax, letter, label, detail=None):
    p = ax.get_position()
    lift = .052 if detail else .023
    fig.text(p.x0 - .041, p.y1 + lift, letter, weight="bold", fontsize=8, va="bottom")
    fig.text(p.x0, p.y1 + lift, label, fontsize=7, va="bottom")
    if detail:
        fig.text(p.x0, p.y1 + .024, detail, fontsize=6.5, va="bottom")


def geometry_audit(fig, name):
    """Measure rendered glyph boxes, including all titles, ticks and legends.

    Colorbar numbers are placed on purpose-built blank strips, not on cells.
    Data-line intersections are tested separately from axis and keyline strokes.
    Data-to-data crossings are legitimate, not removed to make the plot prettier.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    entries, clipped, overlaps, crossings = [], [], [], []
    inactive_ticks = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            low, high = sorted(limits)
            for tick in axis.get_major_ticks():
                if not low - 1e-10 <= tick.get_loc() <= high + 1e-10:
                    inactive_ticks.update([tick.label1, tick.label2])
    for item in fig.findobj(Text):
        if item in inactive_ticks:
            continue
        if not item.get_visible() or not item.get_text().strip():
            continue
        bb = item.get_window_extent(renderer)
        if bb.width <= 0 or bb.height <= 0:
            continue
        entries.append((item, bb))
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width or bb.y1 > fig.bbox.height:
            clipped.append(item.get_text())
    for index, (a, aa) in enumerate(entries):
        for b, bb in entries[index + 1:]:
            area = max(0, min(aa.x1, bb.x1) - max(aa.x0, bb.x0)) * max(0, min(aa.y1, bb.y1) - max(aa.y0, bb.y0))
            if area > .3:
                overlaps.append([a.get_text(), b.get_text()])
    paths = []
    marker_boxes = []
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_visible():
                paths.append(line.get_path().transformed(line.get_transform()))
        for patch in ax.patches:
            if isinstance(patch, FancyArrowPatch):
                paths.append(patch.get_path().transformed(patch.get_transform()))
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                offsets = collection.get_offset_transform().transform(collection.get_offsets())
                sizes = collection.get_sizes()
                radius = np.sqrt(sizes.max()) * fig.dpi / 72 / 2 if len(sizes) else 0
                marker_boxes.extend(Bbox.from_extents(x-radius, y-radius, x+radius, y+radius) for x, y in offsets)
    for item, bb in entries:
        # Matplotlib's native bbox test can treat compound arrow subpaths as
        # connected. Test actual consecutive segments, respecting MOVETO/CLOSE.
        def hits(path):
            previous = None
            for points, code in path.iter_segments(curves=False, simplify=False):
                point = np.asarray(points[-2:])
                if code == MplPath.MOVETO:
                    previous = point
                elif code == MplPath.LINETO and previous is not None:
                    if MplPath([previous, point]).intersects_bbox(bb, filled=False):
                        return True
                    previous = point
                elif code == MplPath.CLOSEPOLY:
                    previous = None
            return False
        if any(hits(path) for path in paths) or any(bb.overlaps(mb) for mb in marker_boxes):
            crossings.append(item.get_text())
    report = {"figure": name, "texts_checked": len(entries), "minimum_font_pt": min(t.get_fontsize() for t, _ in entries),
              "clipped_text": clipped, "text_text_overlaps": overlaps, "text_data_line_or_marker_intersections": crossings}
    GEOMETRY_REPORTS.append(report)
    if clipped or overlaps or crossings:
        fig.savefig(PREVIEW / f"{name}_failed_layout.png", dpi=220, bbox_inches=None)
        print(json.dumps(report, ensure_ascii=False))
        raise RuntimeError(f"Layout gate failed for {name}")
    return report


def save(fig, name, data, description, manuscript_number):
    geometry_audit(fig, name)
    data.to_csv(OUT / f"{name}_source_data.csv", index=False)
    for fmt in ["pdf", "svg", "png"]:
        path = OUT / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=DPI, bbox_inches=None, facecolor="white", transparent=False)
    # Native Agg RGB export: remove alpha without changing any pixel geometry.
    with Image.open(OUT / f"{name}.png") as im:
        assert np.asarray(im.getchannel("A")).min() == 255
        rgb = im.convert("RGB")
        rgb.save(OUT / f"{name}.png", dpi=(DPI, DPI))
    doc = fitz.open(OUT / f"{name}.pdf")
    page = doc[0]
    assert len(page.get_images(full=True)) == 0, "Vector PDF contains a raster"
    page.get_pixmap(dpi=220, alpha=False).save(PREVIEW / f"{name}.png")
    fonts = page.get_fonts(full=True)
    assert fonts and all(f[2] != "Type3" for f in fonts)
    svg = (OUT / f"{name}.svg").read_text(encoding="utf-8")
    assert "<image" not in svg and "<text" in svg
    with Image.open(OUT / f"{name}.png") as im:
        raster = {"pixels": list(im.size), "dpi": list(im.info["dpi"]), "mode": im.mode}
    RECORDS.append({"stem": name, "manuscript_figure": manuscript_number, "width_mm": WIDTH_MM,
                    "height_mm": round(fig.get_figheight() * 25.4, 3), "raster": raster,
                    "vector_raster_objects": 0, "fonts": [f[3] for f in fonts], "rows_in_source_data": len(data),
                    "description": description,
                    "sha256": {fmt: digest(OUT / f"{name}.{fmt}") for fmt in ["pdf", "svg", "png"]}})
    doc.close()
    plt.close(fig)
    print(f"PASS figure {manuscript_number}: {name}", flush=True)


def workflow():
    fig = figure(66)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0,165), ylim=(0,66)); ax.axis("off")
    def box(x,y,w,h,heading,body,accent):
        ax.add_patch(Rectangle((x,y),w,h,facecolor="#F6F7F8",edgecolor="#CED3D7",lw=.6))
        ax.add_patch(Rectangle((x,y+h-1.4),w,1.4,facecolor=accent,edgecolor="none"))
        ax.text(x+3,y+h-5.5,heading,weight="bold",va="top",fontsize=7)
        ax.text(x+3,y+h-11,body,va="top",fontsize=7,linespacing=1.5)
    box(3,17,38,34,"Regional comparison", "127 NUTS-3 regions\nBE 32 / DE-LU 32\nFR 31 / NL 32\nFour shared price series", "#555555")
    box(56,37,48,26,"Energy bounds", "Historical forecasts + PV proxy\nDaily operating model\nCandidate-specific cost intervals", BLUE)
    box(56,4,48,26,"Mapped land context", "CORINE 2018 / Natura 2024\nWhole region + clipped buffers\nFour separate preferences", "#555555")
    box(119,17,43,34,"Exclusion register", "Versioned removal certificates\nNamed retained witnesses\nDependency-based reopening\nReview snapshot; no approval", ORANGE)
    for xy in [[(41,34),(48,34),(48,50),(55,50)],[(41,34),(48,34),(48,17),(55,17)],
               [(104,50),(111,50),(111,40),(118,40)],[(104,17),(111,17),(111,27),(118,27)]]:
        ax.add_patch(FancyArrowPatch(path=MplPath(xy,[1,2,2,2]),arrowstyle="-|>",mutation_scale=7,lw=.8,color="#555555"))
    ax.text(3,8,"Regional options for comparison",fontsize=6.5)
    ax.text(119,8,"Not parcel approval",fontsize=6.5)
    data = pd.DataFrame({"zone":ZONES,"regions":[32,32,31,32]})
    save(fig,"figure1_land_workflow",data,"Cost and context evidence support versioned removal certificates and dependency-based reopening; no parcel approval is implied.",1)


def geo_path(poly):
    poly = orient(poly, sign=1)
    parts = []
    for ring in [poly.exterior, *poly.interiors]:
        xy = np.asarray(ring.coords) / 1e6
        codes = np.full(len(xy), MplPath.LINETO, dtype=np.uint8)
        codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
        parts.append(MplPath(xy,codes))
    return MplPath.make_compound_path(*parts)


def maps(frame):
    features = json.loads(GIS.read_text(encoding="utf-8"))["features"]
    projector = Transformer.from_crs(4326,3035,always_xy=True).transform
    geoms = {f["properties"]["NUTS_ID"]:transform(projector,shape(f["geometry"])) for f in features
             if f["properties"]["CNTR_CODE"] in {"BE","DE","FR","LU","NL"}
             and not f["properties"]["NUTS_ID"].startswith(("FRM","FRY"))}
    indexed = frame.set_index("candidate_id")
    backgrounds, active, ids = [], [], []
    for cid, geom in geoms.items():
        for poly in list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]:
            patch = PathPatch(geo_path(poly))
            backgrounds.append(patch)
            if cid in indexed.index:
                active.append(patch); ids.append(cid)
    # Same sequential map and explicit independent scales for three attributes.
    cmap = colors.LinearSegmentedColormap.from_list("context_blue",["#E5F2FA","#77AFCF","#286C9A","#093C66"])
    fig = figure(91)
    columns = [CONTEXTS[0],CONTEXTS[2],CONTEXTS[3]]
    labels = ["Industrial / commercial", "Forest / semi-natural", "Natura 2000"]
    rows = []
    for i,(col,label) in enumerate(zip(columns,labels)):
        x = .017 + i*.333
        ax = fig.add_axes([x,.245,.305,.645])
        ax.add_collection(PatchCollection(backgrounds,facecolor="#F0F0F0",edgecolor="#D4D4D4",linewidth=.15))
        vmax = np.ceil(indexed[col].max()*100/5)*5
        collection = PatchCollection(active,cmap=cmap,norm=colors.Normalize(0,vmax),edgecolor="#505050",linewidth=.23)
        collection.set_array(np.array([indexed.loc[c,col]*100 for c in ids]))
        ax.add_collection(collection)
        bounds=np.array([g.bounds for g in geoms.values()])/1e6
        ax.set(xlim=(bounds[:,0].min()-.035,bounds[:,2].max()+.035),
               ylim=(bounds[:,1].min()-.035,bounds[:,3].max()+.035),aspect="equal"); ax.axis("off")
        fig.text(x,.943,chr(97+i),weight="bold",fontsize=8)
        fig.text(x+.026,.944,label,fontsize=7)
        cax = fig.add_axes([x+.039,.171,.224,.028])
        cbar = fig.colorbar(collection,cax=cax,orientation="horizontal",ticks=[0,vmax/2,vmax])
        if cbar.solids is not None:
            cbar.solids.set_rasterized(False)
            cbar.solids.set_edgecolor("face")
        cbar.outline.set_linewidth(.45)
        cbar.ax.set_xlabel("Regional area share (%)",labelpad=3)
        for cid in indexed.index:
            rows.append({"candidate_id":cid,"zone":indexed.loc[cid,"zone"],"attribute":col,"area_share_pct":indexed.loc[cid,col]*100,"colour_max_pct":vmax})
    fig.legend(handles=[Rectangle((0,0),1,1,facecolor="#F0F0F0",edgecolor="#B5B5B5",lw=.5)],
               labels=["Outside the 127-region cohort"],loc="lower left",bbox_to_anchor=(.015,.004),frameon=False)
    fig.text(.50,.035,"Not developable land; boundaries © EuroGeographics",fontsize=6.5)
    save(fig,"figure2_land_context_maps",pd.DataFrame(rows),"Three regional choropleths; grey regions are outside the cohort, not missing selected observations. EPSG:3035 equal-area projection.",2)


def four_panels(height=112):
    fig = figure(height)
    axes = [fig.add_axes([x,y,.365,.27]) for y in [.63,.14] for x in [.11,.61]]
    return fig, axes


def frontiers(graph):
    fig,axes = four_panels(115)
    xmin,xmax = 0,np.ceil(graph[CONTEXTS[3]].max()*100/10)*10
    ymin,ymax = np.floor(graph[COST].min()*10)/10-.03,np.ceil(graph[COST].max()*10)/10+.03
    for ax,zone,letter in zip(axes,ZONES,"abcd"):
        local = graph.loc[graph.zone.eq(zone)]
        removed,kept = local.loc[~local.retained],local.loc[local.retained]
        ax.scatter(removed[CONTEXTS[3]]*100,removed[COST],s=13,color="#858585",marker="x",linewidths=.65,zorder=2)
        ax.scatter(kept[CONTEXTS[3]]*100,kept[COST],s=13,color=BLUE,marker="o",linewidths=.35,zorder=3)
        frontier = local.loc[local.frontier]
        ax.scatter(frontier[CONTEXTS[3]]*100,frontier[COST],s=50,facecolors="none",edgecolors=ORANGE,linewidths=.9,zorder=4)
        ax.set(xlim=(-1,xmax+1),ylim=(ymin,ymax),xticks=np.arange(0,xmax+.1,20),
               xlabel="Natura 2000 coverage (%)")
        if zone in ["BE","FR"]: ax.set_ylabel("Electricity cost (2023 EUR/kg)")
        omission = int((local.frontier & ~local.retained).sum())
        title(fig,ax,letter,NAMES[zone],f"n = {len(local)}; frontier omitted {omission}/{int(local.frontier.sum())}")
    handles = [Line2D([],[],color="#858585",marker="x",ls="none",markersize=3.2,label="Removed"),
               Line2D([],[],color=BLUE,marker="o",ls="none",markersize=3.2,label="Retained"),
               Line2D([],[],color=ORANGE,marker="o",mfc="none",ls="none",markersize=5,label="Cost-context frontier")]
    fig.legend(handles=handles,ncol=3,loc="lower center",bbox_to_anchor=(.52,.008),columnspacing=1.5)
    save(fig,"figure3_cost_protection_frontiers",graph[["candidate_id","zone",COST,CONTEXTS[3],"retained","frontier"]],
         "All 127 points on common cost and designation scales. Crosses are removed, circles retained; orange rings identify frontier options.",3)


def profiles(graph):
    fig,axes = four_panels(115)
    x = np.arange(4)
    exported = []
    for ax,zone,letter in zip(axes,ZONES,"abcd"):
        local = graph.loc[graph.zone.eq(zone)].set_index("candidate_id")
        for cid,row in local.iterrows():
            vals = row[CONTEXTS].astype(float).to_numpy()*100
            role = "not_shown_retained_nonfrontier"
            if not row.retained:
                ax.plot(x,vals,color="#CACACA",lw=.45,zorder=1)
                role = "removed"
            if row.frontier:
                if row.retained:
                    ax.plot(x,vals,color=BLUE,lw=.85,ls="--",marker="s",ms=2.2,zorder=3)
                    role = "retained_frontier"
                else:
                    ax.plot(x,vals,color=ORANGE,lw=1,marker="o",ms=2.4,zorder=4)
                    role = "removed_frontier"
            for attribute,value in zip(CONTEXTS,vals):
                exported.append({"candidate_id":cid,"zone":zone,"attribute":attribute,"area_share_pct":value,"visual_role":role})
        ax.set(xlim=(-.18,3.18),ylim=(-3,103),xticks=x,xticklabels=CONTEXT_LABELS,yticks=[0,25,50,75,100])
        ax.tick_params(axis="x",length=0,pad=5,labelsize=6.5)
        if zone in ["BE","FR"]: ax.set_ylabel("Mapped area share (%)")
        title(fig,ax,letter,NAMES[zone])
    handles = [Line2D([],[],color="#AAAAAA",lw=.7,label="Other removed regions"),
               Line2D([],[],color=ORANGE,lw=1,marker="o",ms=3,label="Removed frontier"),
               Line2D([],[],color=BLUE,lw=.9,ls="--",marker="s",ms=2.6,label="Retained frontier")]
    fig.legend(handles=handles,ncol=3,loc="lower center",bbox_to_anchor=(.52,.010),columnspacing=1.05)
    save(fig,"figure6_excluded_land_context_profiles",pd.DataFrame(exported),
         "Parallel-coordinate profiles use raw shares on the same 0-100% scale; Natura overlaps land-cover classes. Lines connect attributes, not spatial neighbours.",4)


def text_contrast(rgba):
    rgb = np.array(rgba[:3])
    linear = np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    lum = linear @ np.array([.2126,.7152,.0722])
    return "white" if (1.05/(lum+.05)) > ((lum+.05)/.05) else "black"


def matrix_panel(fig, ax, grid, labels, xticks, yticks, cmap, xlabel):
    # pcolormesh, unlike imshow, creates genuine vector cells.
    cm = matplotlib.colormaps[cmap]
    ax.pcolormesh(np.arange(5)-.5,np.arange(5)-.5,grid,cmap=cm,vmin=0,vmax=1,edgecolors="white",linewidth=1.5,rasterized=False)
    ax.set(xlim=(-.5,3.5),ylim=(3.5,-.5),xticks=range(4),xticklabels=xticks,yticks=range(4),yticklabels=yticks)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,labels[i][j],ha="center",va="center",fontsize=7,color=text_contrast(cm(grid[i,j])))
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0,pad=5)
    ax.set_xlabel(xlabel,labelpad=7)
    # All heatmaps use the same explicit fraction scale; labels sit outside data.
    p = ax.get_position()
    cax = fig.add_axes([p.x0,p.y0-.185,p.width,.028])
    mappable = matplotlib.cm.ScalarMappable(norm=colors.Normalize(0,1),cmap=cm)
    bar = fig.colorbar(mappable,cax=cax,orientation="horizontal",ticks=[0,.5,1])
    bar.ax.set_xticklabels(["0","0.5","1"])
    bar.set_label("Fraction",labelpad=2)
    bar.outline.set_linewidth(.4)
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")


def omission_retention(comparison):
    primary = comparison.loc[comparison.scenario.eq("u50_fourweek") & comparison.scale.eq("nuts3")]
    fig = figure(92)
    axes = [fig.add_axes([.125,.40,.35,.43]),fig.add_axes([.625,.40,.35,.43])]
    records=[]
    for ax,selector,letter,label,cmap in zip(axes,["electricity_certificate","land_bound_extrema_anchors"],"ab",
                                           ["Frontier omissions / full frontier","Retained regions / cohort"],["Oranges","Blues"]):
        grid=np.zeros((4,4)); annotations=[]
        for i,zone in enumerate(ZONES):
            rowlabels=[]
            for j,view in enumerate(VIEWS):
                row=primary.loc[primary.zone.eq(zone)&primary.view.eq(view)&primary.selector.eq(selector)].iloc[0]
                numerator,denominator = (int(row.frontier_omitted),int(row.frontier_size)) if selector=="electricity_certificate" else (int(row.retained),int(row.candidate_count))
                grid[i,j]=numerator/denominator
                rowlabels.append(f"{numerator}/{denominator}")
                records.append({"zone":zone,"view":view,"selector":selector,"numerator":numerator,"denominator":denominator,"fraction":grid[i,j]})
            annotations.append(rowlabels)
        matrix_panel(fig,ax,grid,annotations,["Industrial","Agri-\ncultural","Forest /\nsemi-natural","Natura\n2000"],
                     ["BE","DE-LU","FR","NL"],cmap,"Land-context preference")
        title(fig,ax,letter,label)
    fig.text(.125,.955,"Primary model: 50% utilisation, four-week forecast, whole-region support",fontsize=7)
    fig.text(.125,.025,"Counts refer to regions; all land-aware frontiers are retained.",fontsize=6.5)
    save(fig,"figure4_omission_retention_tradeoff",pd.DataFrame(records),
         "Left: electricity-only omission counts. Right: context-aware retention counts. Both panels have a 0-1 colour scale and explicit denominators.",5)


def sensitivity(comparison):
    fig=figure(94)
    axes=[fig.add_axes([.205,.38,.285,.44]),fig.add_axes([.705,.38,.285,.44])]
    records=[]
    for ax,selector,letter,label,cmap in zip(axes,["electricity_certificate","land_bound_extrema_anchors"],"ab",
                                           ["Frontier omissions / full frontier","Retained regions / cohort"],["Oranges","Blues"]):
        grid=np.zeros((4,4)); annotations=[]
        for i,scenario in enumerate(SCENARIOS):
            rowlabels=[]
            for j,scale in enumerate(SCALES):
                rows=comparison.loc[comparison.scenario.eq(scenario)&comparison.scale.eq(scale)&comparison.view.eq("protected_context")&comparison.selector.eq(selector)]
                assert len(rows)==4
                numerator,denominator=(int(rows.frontier_omitted.sum()),int(rows.frontier_size.sum())) if selector=="electricity_certificate" else (int(rows.retained.sum()),int(rows.candidate_count.sum()))
                grid[i,j]=numerator/denominator;rowlabels.append(f"{numerator}/{denominator}")
                records.append({"scenario":scenario,"scale":scale,"selector":selector,"numerator":numerator,"denominator":denominator,"fraction":grid[i,j]})
            annotations.append(rowlabels)
        matrix_panel(fig,ax,grid,annotations,["NUTS-3","20 km","10 km","5 km"],SCENARIO_LABELS,cmap,"Spatial support")
        ax.tick_params(axis="y",labelsize=6.5)
        title(fig,ax,letter,label)
    fig.text(.205,.96,"Natura 2000 preference: four model assumptions × four spatial supports",fontsize=7)
    fig.text(.02,.885,"Utilisation / forecast",fontsize=6.5)
    fig.text(.52,.885,"Utilisation / forecast",fontsize=6.5)
    fig.text(.205,.025,"Buffers are clipped to each parent region; they are not surveyed parcels.",fontsize=6.5)
    save(fig,"figure5_scale_model_sensitivity",pd.DataFrame(records),
         "Four utilisation/forecast assumptions and four supports; protected-context omission persists in all 16 comparisons, with support-dependent denominators.",6)


def contact_sheet():
    thumbs=[]
    for r in sorted(RECORDS,key=lambda x:x["manuscript_figure"]):
        with Image.open(PREVIEW / f"{r['stem']}.png") as im:
            im=im.convert("RGB");im.thumbnail((790,590));thumbs.append(im.copy())
    sheet=Image.new("RGB",(1640,1860),"#EEEEEE")
    for i,im in enumerate(thumbs):
        sheet.paste(im,((i%2)*820+(820-im.width)//2,(i//2)*620+(620-im.height)//2))
    sheet.save(PREVIEW / "six_figures_contact.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser();parser.add_argument("--only",choices=["workflow","maps","frontiers","profiles","tradeoff","sensitivity"])
    args=parser.parse_args()
    for directory in [OUT,QA,PREVIEW]:directory.mkdir(parents=True,exist_ok=True)
    configure()
    manifest=json.loads((RESULT/"analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["land_candidate_count"]==127 and not manifest["preview_only"]
    land=pd.read_csv(LAND)
    land=land.loc[land.scale.eq("nuts3") & ~land.candidate_id.str.startswith("FRM")]
    energy=pd.read_csv(ENERGY)
    energy=energy.loc[~energy.candidate_id.str.startswith("FRM")]
    frame=energy.merge(land.drop(columns=["zone"]),on="candidate_id",validate="one_to_one")
    assert len(frame)==127 and frame[CONTEXTS+[COST]].notna().all().all()
    members=pd.read_csv(RESULT/"land_shortlist_membership.csv")
    primary=members.loc[members.scenario.eq("u50_fourweek") & members.scale.eq("nuts3") & members.view.eq("protected_context") & members.selector.eq("electricity_certificate")]
    graph=frame.merge(primary[["candidate_id","retained","frontier"]],on="candidate_id",validate="one_to_one")
    assert graph.retained.sum()==48 and graph.frontier.sum()==19 and (graph.frontier & ~graph.retained).sum()==8
    comparison=pd.read_csv(RESULT/"land_shortlist_comparison.csv")
    functions={"workflow":workflow,"maps":lambda:maps(frame),"frontiers":lambda:frontiers(graph),"profiles":lambda:profiles(graph),
               "tradeoff":lambda:omission_retention(comparison),"sensitivity":lambda:sensitivity(comparison)}
    for key,fn in functions.items():
        if args.only is None or args.only==key:fn()
    payload={"design":"Nature-inspired artwork for a Land research article, not Nature submission certification.",
             "master_width_mm":WIDTH_MM,"native_raster_dpi":DPI,"matplotlib_version":matplotlib.__version__,"python":platform.python_version(),
             "script_sha256":digest(Path(__file__)),"upstream_result_manifest_sha256":digest(RESULT/"analysis_manifest.json"),
             "source_hashes":{str(p):digest(p) for p in [LAND,ENERGY,GIS,RESULT/"land_shortlist_membership.csv",RESULT/"land_shortlist_comparison.csv"]},
             "reviewed_github_resources":{"SciencePlots":{"repository":"https://github.com/garrettj403/SciencePlots","commit":"b9b16959570bd2fbc9ff5118bacc423c3bddd592"},
                                         "scientific-visualization":{"repository":"https://github.com/K-Dense-AI/scientific-agent-skills","commit":"154988403bb5a18e9d3c0ce4e6d5e2e4b184a298"}},
             "raw_resource_hashes":{p.name:digest(p) for p in RAW.iterdir() if p.is_file()},
             "geometry_checks":GEOMETRY_REPORTS,"figures":RECORDS,
             "limitations":"Glyph boxes, paths and marker boxes were screened; this is supplemented by manual inspection, not proof against every possible renderer defect. SVG text requires Arial on the viewing/editing system."}
    (QA/"artwork_manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not args.only:contact_sheet()


if __name__=="__main__":main()
