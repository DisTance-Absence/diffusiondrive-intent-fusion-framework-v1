"""Evidence-backed, reviewable road taxonomy; metadata never enters fusion.

Labels below are an assistant review of current HD-map geometry and saved
longitudinal topology, not human annotation or certified driving permissions.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
INVENTORY=ROOT/'scene-choice-inventory-v3/inventory.json'
MANIFEST=ROOT/'joystick-experiment-copy/rule-joystick-contrast-v2/results/frozen/screen_manifest_frozen.json'
CATEGORIES={
    'crossing_junction':'交叉路口',
    'branch_junction':'丁字/分岔路口',
    'roundabout':'环岛式道路',
    'curved_segment':'一般弯道路段',
    'straight_approach':'宽幅直路/路口接近段',
}
# (category, short label, secondary features, geometry evidence, uncertainty, research use)
REVIEWS={
 'S01':('crossing_junction','四臂交叉路口',['四臂交叉','路口接近'],
        '宽视野可见当前进口、前方延续及左右横向支路；已导出中心线在路口处分开。',
        '道路支路形状不代表当前车道允许所有转向；已导出路径不是全部合法走法。',
        '用于检查左右/前向指令与候选覆盖是否匹配。'),
 'S02':('crossing_junction','不规则四臂路口',['斜交/不规则四臂','近路口连接处'],
        '道路可行驶区域呈四臂连接，前向支路斜接，左右支路不对称。',
        '部分支路在导出地图中较短，后续连通与转向许可待查。',
        '用于不规则路口中的方向指令匹配。'),
 'S03':('crossing_junction','斜交四臂路口',['斜交四臂','路口接近'],
        '纵向道路与斜向横穿道路相交，可见四条道路臂。',
        '具体车道线、车道数及各转向许可尚未核验。',
        '用于斜交道路中指令角度与道路方向的区别分析。'),
 'S04':('crossing_junction','十字交叉路口',['四臂交叉','横向道路延伸'],
        '扩大视野可见左右横向道路及前后纵向连接，构成交叉路口。',
        '原近景图偏向右侧；可见道路范围与候选当前可行范围应分别判断。',
        '用于继续研究anchor精修与最终方向响应差异。'),
 'S05':('roundabout','环岛出口/分流段',['中央圆形岛','环形道路','出口/连接附近'],
        '更宽的可行驶区域显示中央圆形岛及环形道路，当前位置附近连接外部斜向道路。',
        '采用几何意义的环岛式标签；让行、环行方向和出口通行规则未认证。',
        '用于环行与驶出方向意图的候选覆盖研究。'),
 'S06':('branch_junction','渠化斜丁字路口',['斜向主路','中央分隔','进口接入'],
        '当前进口接入一条斜向分隔道路，局部呈丁字/分岔连接。',
        '起始车道匹配偏移约1.42米，接近原匹配阈值；需要重点复核自车所属车道。',
        '用于分岔选择，但应先复核自车车道匹配。'),
 'S07':('branch_junction','带导流岛的侧向岔口',['侧向支路','三角导流岛','连接段'],
        '主路旁接入侧向道路，连接处有三角形非行驶岛；宽视野没有形成近端四臂交叉。',
        '起始匹配为连接段；导流岛两侧的通行方向和优先权未认证。',
        '用于侧向岔路与继续沿主路的方向选择。'),
 'S08':('crossing_junction','分隔式斜交路口',['四臂斜交','中央分隔','左向支路'],
        '宽视野除近景可见的左侧支路外，还能看到右侧接入道路，整体为四臂斜交结构。',
        '不能根据旧近景把它判为丁字路口；道路存在支路不保证候选覆盖或合法可达。',
        '用于模型偏好与外部直行/左向意图的融合对照。'),
 'S09':('branch_junction','右侧丁字/分岔路口',['右侧支路','支路中央分隔','主路延续'],
        '当前道路向前延续，右侧有一条带中央分隔的侧向道路接入。',
        '侧向连接的车道边界与转向许可待查。',
        '用于前向与右向意图选择。'),
 'S10':('crossing_junction','宽幅分隔式十字路口',['大型交叉口','中央分隔','宽幅道路'],
        '扩大视野后左右横向道路均可见，与当前纵向分隔道路构成大型四臂路口。',
        '原近景仅突出右转连接；道路宽度不能直接换算成可变道数量。',
        '用于宽幅路口中的前向/右向选择及环境拒绝分析。'),
 'S11':('branch_junction','丁字路口/弯曲主路接入',['左右分流','弯曲主路','进口终接'],
        '当前进口前方接入横向道路，向左延伸，向右沿弯曲道路延续，呈丁字连接。',
        '更远处另有道路连接，当前标签描述近端进口，不代表整个地图仅有三臂。',
        '用于左右分支意图和当前候选支持不足的分析。'),
 'S12':('curved_segment','缓弯道路段',['轻微连续弯曲','当前纵向车道延续','远处侧向连接'],
        '自车前方近端道路及中心线有连续轻微弯曲；当前截取路径沿同一LANE延续。',
        '更远处有侧向连接；单条已导出路径不能证明该路只有一条车道。',
        '用于道路跟随和不必要方向干预的对照。'),
 'S13':('straight_approach','宽幅直路接近路口',['近端平直','分隔式宽路','前方交叉口','变道潜力待核验'],
        '当前近端道路平直且较宽；扩大视野可见前方大型交叉口，周边参与者呈多列分布。',
        '缺少左右相邻车道及实虚线信息，不能认证可变道；也不能标为远离路口的纯直路。',
        '优先补查相邻车道与跨线许可，再决定是否作为变道实验场景。'),
 'S14':('straight_approach','宽幅直路接近连接口',['近端平直','中央分隔','前方侧向连接','变道潜力待核验'],
        '当前位于较宽的纵向直路段，前方有带导流岛的侧向连接。',
        '缺少相邻车道边界及跨线许可；多列车辆只提示多车道可能，不能单独认证变道。',
        '优先补查相邻车道与交通条件，可作为变道候选场景复核。'),
 'S15':('roundabout','环岛内弯曲连接段',['中央圆形岛','环形弯道','当前连接段'],
        '宽视野可见中央圆形岛及围绕它的环形可行驶带；自车位于环形道路弯曲连接处。',
        '只导出一条后继路径不代表全环岛只有一种选择；当前位置邻近开口是否可驶入需另查。',
        '用于环形道路跟随与误把道路曲率当外部转向的分析。'),
 'S16':('crossing_junction','斜交四臂路口',['斜交四臂','道路宽度不对称','路口接近'],
        '纵向道路与斜向横穿道路在前方交会，宽视野可见四臂。',
        '道路支路几何标签不代表精修候选已覆盖每个出口。',
        '用于左/前/右方向覆盖与意图响应的对照。'),
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def classify():
    inventory=json.loads(INVENTORY.read_text(encoding='utf-8'))
    manifest={s['token']:s for s in json.loads(MANIFEST.read_text(encoding='utf-8'))['scenes']}
    records=[]
    for scene in inventory['scenes']:
        sid=scene['scene_id'];source=manifest[scene['token']]
        category,label,features,geometry,uncertainty,use=REVIEWS[sid]
        origin='车道连接段 LANE_CONNECTOR' if source['start_layer']=='LANE_CONNECTOR' else '普通车道 LANE'
        evidence=[geometry,
           f"历史地图匹配：{origin}，ID {source['start_lane_id']}，中心线偏移 {source['start_lane_offset_m']:.3f} m。",
           f"当前保存 {len(scene['map_paths'])} 条纵向连接路径；数量不等于车道数，也不是完整可选行为枚举。"]
        notes=[uncertainty,'本分类依据地图几何和历史拓扑复核，尚未经人工标注确认。']
        lane_note=('可作为变道候选场景复核，尚缺相邻车道、车道线及通行许可证据。'
                    if sid in ('S13','S14') else '本轮按道路结构分类，未核验合法变道条件。')
        records.append(dict(scene_id=sid,token=scene['token'],primary_type=category,
           primary_label=CATEGORIES[category],short_label=label,secondary_features=features,
           evidence=evidence,uncertainties=notes,status='provisional_map_review',
           confidence='medium' if sid in ('S02','S06','S07') else 'high',
           confidence_scope='broad geometric road type only; not traffic permission or human annotation',
           lane_change_status='unverified',lane_change_candidate=sid in ('S13','S14'),
           lane_change_note=lane_note,variable_direction_lane_status='unverified',
           context_image=f'figures/classification/{sid}_context.png',research_use=use,
           source_evidence=dict(start_layer=source['start_layer'],start_lane_id=source['start_lane_id'],
                start_lane_offset_m=source['start_lane_offset_m'],path_ids=[p['path_ids'] for p in scene['map_paths']],
                basis=['current_drivable_area_geometry','saved_longitudinal_map_paths','historical_map_match'],
                not_used=['candidate_directions','model_scores','NC_DAC_TTC','fusion_results'])))
    counts=Counter(r['primary_type'] for r in records)
    meta=dict(date='2026-10-07',version='road_structure_review_v1',
       categories=[dict(id=k,label=v,count=counts[k]) for k,v in CATEGORIES.items()],
       scope='按自车当前近端道路结构分类，另注前方路口与连接阶段；同一路口中只导出部分车道后继，不影响道路结构标签。',
       note='地图几何复核初分，不代表人工确认、合法变道认证或候选行为覆盖。分类仅用于浏览与分组，不进入评分或环境检查。',
       terminology='可变道与可变导向/潮汐车道分别记录；当前两者均未认证。S13、S14仅列为变道潜力待核验。',
       review_type='assistant_map_geometry_and_topology_review_not_human_annotation',
       cohort_note='本批主要是路口与环岛相关场景，不能当作道路类型均衡或已覆盖合法变道的测试集。',
       overview_image='figures/classification/overview.png')
    document=dict(meta=meta,scenes=records,sources=[dict(path=str(p),sha256=sha(p)) for p in (INVENTORY,MANIFEST)],
         method='Broader road-only map review plus map-match/topology evidence; four-second candidate endpoints are excluded from taxonomy.')
    save(BASE/'scene_classification.json',document)
    return document,inventory

def render(document, inventory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PatchPolygon
    from shapely.geometry import Polygon,box
    from shapely.ops import unary_union
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    folder=BASE/'figures/classification';folder.mkdir(parents=True,exist_ok=True)
    lookup={r['scene_id']:r for r in document['scenes']}
    def draw(ax,scene,compact=False):
        clipped=unary_union([Polygon(p['exterior'],p.get('holes',[])) for p in scene['road_context']['drivable_polygons_ego']]).intersection(box(-60,-80,80,80))
        parts=[clipped] if clipped.geom_type=='Polygon' else list(clipped.geoms)
        ax.set_facecolor('#e4e9ee')
        for poly in parts:
            if poly.geom_type!='Polygon':continue
            ax.add_patch(PatchPolygon([(y,x) for x,y in poly.exterior.coords],facecolor='white',edgecolor='#a0adba',lw=.7))
            for ring in poly.interiors:
                ax.add_patch(PatchPolygon([(y,x) for x,y in ring.coords],facecolor='#e4e9ee',edgecolor='#a0adba',lw=.7))
        for path in scene['map_paths']:
            pts=path['centerline_xy'];ax.plot([p[1] for p in pts],[p[0] for p in pts],color='#497fa7',lw=1)
        v=scene['road_context']['vehicle_parameters'];w=v['width']/2
        ax.add_patch(PatchPolygon([(-w,-v['rear_length']),(-w,v['front_length']),(w,v['front_length']),(w,-v['rear_length'])],facecolor='#243d57'))
        ax.plot(0,7,'^',color='#243d57',markersize=6)
        if not compact:ax.annotate('当前自车',xy=(0,0),xytext=(-13,-14),arrowprops=dict(arrowstyle='->',color='#243d57'),fontsize=10)
        ax.set_xlim(80,-80);ax.set_ylim(-60,80);ax.set_aspect('equal');ax.grid(alpha=.18);ax.tick_params(labelsize=7 if compact else 9)
        ax.set_title(scene['scene_id']+' · '+lookup[scene['scene_id']]['short_label'],fontsize=10 if compact else 15)
        ax.set_xlabel('横向 y / m（左正）',fontsize=8 if compact else 10);ax.set_ylabel('前向 x / m',fontsize=8 if compact else 10)
    fig,axes=plt.subplots(4,4,figsize=(20,18))
    for ax,scene in zip(axes.flat,inventory['scenes']):draw(ax,scene,True)
    fig.suptitle('16 场景道路结构分类｜白色为地图可行驶区，蓝线为已导出地图中心线；不含模型候选',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.975));fig.savefig(folder/'overview.png',dpi=130);plt.close(fig)
    for scene in inventory['scenes']:
        fig,ax=plt.subplots(figsize=(10,9));draw(ax,scene)
        fig.text(.5,.014,'地图几何初分，待人工确认；中心线数量不等于车道数，未认证变道或可变导向规则。',ha='center',fontsize=10,color='#516779')
        fig.tight_layout(rect=(0,.035,1,1));fig.savefig(folder/f"{scene['scene_id']}_context.png",dpi=110);plt.close(fig)

def attach(document):
    path=BASE/'data.json';data=json.loads(path.read_text(encoding='utf-8'))
    lookup={r['scene_id']:r for r in document['scenes']}
    assert set(lookup)=={s['id'] for s in data['scenes']}
    for scene in data['scenes']:
        r=lookup[scene['id']];assert r['token']==scene['token']
        scene['classification']={k:v for k,v in r.items() if k not in ('scene_id','token')}
    data['meta']['classification']=document['meta']
    path.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    provenance=BASE/'source_provenance.json';p=json.loads(provenance.read_text(encoding='utf-8'))
    p['road_classification']=dict(path=str(BASE/'scene_classification.json'),sha256=sha(BASE/'scene_classification.json'),
         metadata_only=True,date='2026-10-07',sources=document['sources'])
    for c in p['code']:
        c['sha256']=sha(Path(c['path']))
    p['road_classification']['builder']=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__)))
    save(provenance,p)

if __name__=='__main__':
    document,inventory=classify();render(document,inventory);attach(document)
    print(json.dumps(dict(scenes=len(document['scenes']),categories=document['meta']['categories']),ensure_ascii=False))
