# ROS hNNG reference vs HNNG.jl port spec: review

Reviewer: teammate `ros-review`, 2026-10-16. Read-only. Nothing was built or run.

## 0. Sources and citation keys

| Key | Path |
|---|---|
| VC | `/home/coral/working/archives/race/raceua/race_view_clustering/src/view_clustering.cpp` |
| VH | `.../race_view_clustering/src/view_clustering.h` |
| VU | `.../race_view_clustering/src/view_clustering_utils.cpp` |
| VUH | `.../race_view_clustering/src/view_clustering_utils.h` |
| VM | `.../race_view_clustering/src/view_clustering_manager.cpp` |
| MSG/x | `.../race_view_clustering/msg/x.msg` (PerceptualElement, PerceptualCluster, ObjectRepresentation) |
| SRV/x | `.../race_view_clustering/srv/x.srv` |
| PARAMS | `.../race_view_clustering/params/default_params.yaml` |
| SITOV, RTOV | `/home/coral/working/archives/race/race/race_object_perception/race_perception_msgs/msg/{SITOV,RTOV}.msg` |
| PMH | `.../race_perception_msgs/include/race_perception_msgs/perception_msgs.h` |
| SI | `/home/coral/working/archives/race/race/race_object_perception/race_feature_extraction/src/spin_image.cpp` |
| PDB | `/home/coral/working/archives/race/race/race_object_perception/race_perception_db/src/perception_db.cpp` |
| SIM | `/home/coral/working/archives/race/raceua/race_perception_simulator/scripts/teaching_simulator.py` |
| SPEC | `/home/coral/working/HNNG.jl/docs/superpowers/specs/2026-10-15-hnng-jl-port-design.md` |
| BRIEF | `/home/coral/working/.workflow/scratch/port/cpp-algorithm-brief.md` |

`launch/bringup.launch` only loads PARAMS into a node or nodelet (bringup.launch:13, :21). `mainpage.dox` is
an empty template (mainpage.dox:5-14). Neither adds behaviour.

Notation used below. "Live call path" means code reachable from the three advertised services and the
label subscriber (VM:140-151). Code that is commented out or has no caller is marked "dead".

---------------------------------------------------------------------------------------------------
## 1. Element and distance

**What an element is.** An element is one *view* of a tracked object (an RTOV). It is keyed
`ELEM_<level>_<track_id>_<view_id>`:
- MSG/PerceptualElement:4 `### Key: ELEM_<level>_<track_id>_<view_id>`; fields `view_key`, `outlink_key`,
  `outlink_distance`, `inlink_keys`, `object_name` (MSG/PerceptualElement:11-15).
- The key is built by `PDB:546-554` as `k_name + level + SEP + tid + SEP + vid`, with `EL = "ELEM_"`
  (PMH:34) and `SEP = "_"` (PMH:69). The key is a **decimal string**. Element and cluster maps are
  `std::map<string,...>` (VUH:41-43, `typedef map<string, PerceptualElement> elementMap;`), so
  iteration order is lexicographic on these strings (`"10" < "9"`). This matters for ties (§4, §6).
- A view holds a variable number of spin images: RTOV:11 `string[] sitov_keys  #list of spin image key (SITOV)`.
  Each spin image has a fixed length: SITOV:12 `float32[] spin_image    ## spin image (153 value).`
  So an element is a **variable-size set of 153-d vectors**.
- Views per object: one teaching event inserts every stored view of a track:
  VC:53 `vector<string> added_rtovs = _pdb->getRTOVs(track_id);` then VC:56-63 pushes one element per RTOV.
  The count is whatever the tracker stored for that track. It is not fixed. VC:43-46 ignores `view_id`:
  `"View id != -1. This behaviour is not implemented. I will use as if its equal to -1"`.
- An upper-level element is a copy that shares the view: VU:362-371 `newParentElement` sets
  `pe->level = elem.level+1; pe->view_key = elem.view_key; pe->object_name = elem.object_name;`.

**Distance.** There is one distance function. It has no KL divergence, no chi-square and no sqrt anywhere
in the package (a grep for kullback/chi/divergence/sqrt/hellinger/bhattacharyya over `src/` matches nothing).
1. Spin image to spin image: **squared Euclidean**, with no square root.
   SI:252-255 `for (size_t j = 0; ...) difference += pow( sp1.spin_image.at(j) - sp2.spin_image.at(j), 2.0);`
   If the two lengths differ, the function returns 0 (error) (SI:247-250).
2. View to view, directed: the **mean over A's spin images of the min over B** (comment at SI:266
   `//D(t,O)=(Sigma (min D(tl,Ok)))/q`). Code: SI:279-312. The per-row minimum starts at the constant
   `float minimum_distance = 1000000;` (SI:281), so any single difference above 1e6 is capped. On a
   spin-image size mismatch, the function returns early with a partial, undivided sum (SI:293-296).
3. Symmetrised by **max**: VU:1957-1969 `getDistBetweenViews(src, tar)` computes
   `differenceBetweenObjectViews(src_view, tar_view, d1); differenceBetweenObjectViews(tar_view, src_view, d2); if (d1 > d2) dist = d1; else dist = d2;`.
   The key-based overload short-circuits identical keys to 0 (VU:1980-1983).

The result is the modified Hausdorff distance (Dubuisson–Jain form: max of the two directed mean-of-min
distances) taken over **squared** L2.

**Is it a metric?** No. (a) Squared L2 does not satisfy the triangle inequality. (b) The mean-of-min
(modified Hausdorff) construction does not satisfy it even over plain L2. (c) `d(A,B)=0` whenever A and B
have the same support, so it is at most a pseudo-distance. The 1e6 cap (SI:281) is a further
non-metric artefact. The ROS code nevertheless prunes with radii as if the triangle inequality held (§5,
§7), so its exactness was never guaranteed. This is inference from the formula; nothing in the code
discusses it.

**Where it is computed.** `ViewClusteringUtils::getDistBetweenViews` (VU:1957, :1971, :1978) is the only
entry point. It calls `differenceBetweenObjectViews` (SI:261-316) and `differenceBetweenSpinImage`
(SI:240-259). Views are re-fetched from the perception DB on every call: VU:1984-1985
`vector<SITOV> src_view = _pdb->getSITOVs(source_key); vector<SITOV> tar_view = _pdb->getSITOVs(target_key);`.

**Spec comparison.** SPEC:18-21 (any object, any distance) and SPEC:144-151 (`UserMetric(f; is_metric=false)`
gives an exact but unpruned index) cover this element type. SPEC:422-424 already tests "a non-metric
mean-closest-descriptor distance (`is_metric=false`)", which is the same family. The exact ROS formula
(max-symmetrised, squared L2 per descriptor) is not named. See delta D1.

---------------------------------------------------------------------------------------------------
## 2. Insertion of one element at level 0

Driver: `ViewClustering::addKeyViewsintoPH` (VC:35-237). This is the live path from the label
subscriber, VM:188 `objectHierarchy.data = _viewClustering->addKeyViewsintoPH(msg_in.category_name, msg_in.track_id, msg_in.view_id);`.
The views of the track go into `addList` (VC:56-63). Then a `do { ... } while (clusters_at_level->size()>1);`
loop runs over levels (VC:66-228). Per element of `addList` at a level (VC:74-113):

1. **A.1 add.** VC:77 `_vcUtils->addElement(elements_at_level, addList.at(i));`. A key that is already
   present only produces a warning (VU:399-400 `if(!result.second) ROS_WARN("addElement: redundant element %s"...)`),
   and the existing element is then re-linked. Teaching the same track twice therefore re-inserts its
   views, and `updateLinks` can push a duplicate inlink key (VC:383). Unexpected; recorded.
2. **A.2 nearest neighbour = hierarchical descent, not brute force.** VC:87 `this->getNNElement(...)`.
   `getNNElement` hard-codes the hierarchical path: VC:283 `bool _bknn = true;`. It calls VC:300
   `_vcUtils->getkNNElementwithPH(element.level, src_key, dists, k, true);` with `k = 1` (VC:290). The
   brute-force alternative `getNNElementExhs` (VC:321-370) is reachable only when `_bknn` is false, so it
   is dead.
   `getkNNElementwithPH` (VU:1989-2219) is a top-down **k-count sweep**:
   - The start list is every element at the root level (VU:2005-2007, `check_mod` true).
   - At each level, every element `e` emits three events: `d-r` (DMR), `d` (DEI, or CHN when its child
     cluster's `c_status` is not NOC), and `d+r` (DPR), where `r` is the radius of `e`'s child cluster
     (VU:2040-2046). The events are sorted by value only (VU:2050; the comparator at VUH:68-71 is
     `return (dist < ds.dist);`, and `std::sort` is unstable).
   - A counter `cnt` starts at k (VU:2019). It drops by 1 at each DEI/CHN and by |child members|−1 at
     each DPR (VU:2084 `cnt-= ch_clst->member_keys.size()-1;`). A DMR event keeps its cluster only
     `if (cnt > 0)` (VU:2076-2078).
   - So a child cluster is kept iff its lower bound `d−r` comes before the point where k level-0 points
     are certified. **The sign is correct.**
   - When `check_mod` is set, the members of every changed (non-NOC, non-DEL) child cluster are added
     unconditionally (VU:2146-2164). This compensates for radii that are stale during an update.
   - At the target level, the candidates are scanned exhaustively and sorted (VU:2189-2213). The element
     itself is excluded by `view_key` (VU:2193).
3. **A.3 links.** `updateLinks` (VC:372-401):
   VC:381-383 `element.outlink_key = _vcUtils->getElementKey(nn); element.outlink_distance = dist; nn.inlink_keys.push_back(elem_key);`.
   The element is then erased from the inlinks of its old outlink target (VC:385-397).
4. **A.4 join NN's cluster.** `addElementintoNNCluster` (VC:438-472). A new element is appended to
   `nn_clst->member_keys` (VC:456), and the representative and radius of that cluster are recomputed
   (VC:457).
5. **A.5 rewire sweep.** `updateTheOtherElements` (VC:526-584). The sweep is **level-local**, not
   hierarchical. Its candidate set is bounded by `getNNClusterswithRadius` (VU:2310-2345):
   - Compute `d(e, rep(C))` for **every cluster at this level** (VU:2321-2333).
   - Keep C iff it is the nearest-rep cluster or
     VU:2338 `if (nn_clst == &it->second || dists.at(i)-it->second.radius < min) {`.
     Here `min` is the distance to the nearest rep.
   - Every member of a kept cluster is tested with
     VC:561 `if (dist != 0.0 && dist < other_elem->outlink_distance)`.
   - **This bound is not sound for rewiring.** A member `x` of a pruned cluster needs a rewire iff
     `d(e,x) < outdist(x)`. Only `d(e,x) ≥ d(e,rep) − r` is known, and the code compares that with
     `min` (the NN distance of e), not with x's own outlink distance. Rewires can therefore be missed,
     and the level graph can stop being an exact 1-NN graph. This is the same defect that the C++ fixed
     in fe7641f (BRIEF:174-176, "Before fe7641f the condition was only the first clause, which missed
     members of NN-pruned clusters"). The spec's `maxout` rule (SPEC:266-268) is the fix.
   - `dist != 0.0` means an **exact duplicate is never rewired to** (VC:561). The strict `<` keeps the
     old link on a tie. Neither rule uses ids.
   - A brute-force variant `updateTheOtherElementsExhs` (VC:474-524) exists but is commented out at
     VC:110. It is dead.
6. **Split/merge after a rewire.** `updateCluster` (VC:587-669) recomputes connected components by BFS
   over outlinks and inlinks. `getAllLinks` (VU:602-640) pushes `outlink_key` and `inlink_keys` and
   returns the members **sorted lexicographically** (VU:628-629).
   - VC:616 recomputes the old cluster as the component of the old NN target.
   - VC:620 computes the component of the new target.
   - If the two components are equal and the clusters differ, it is a merge, and the duplicate cluster is
     cleared and deleted (VC:622-634).
   - If the components differ and it was one cluster, it is a split, and a new cluster is created for the
     new target (VC:639-652).
   - Then VC:660-664 assigns the members and re-elects the representative and radius.
   This is "repair affected clusters only", which matches SPEC:272-276 in spirit.

Potential defect: `getAllLinks` pushes `outlink_key` even when it is empty (VU:621), and then calls
`getElement(level, "")`. That returns NULL, which is dereferenced at VU:619-621. This is reachable only
for an element with no outlink. It is not exercised on the normal path and is noted as a risk.

**Spec comparison.** The spec's `link!` (SPEC:264-279) does the NN search and the rewire candidate
recording **in one pruned descent**, with the sound `maxout` bound and canonical ties. ROS does a
hierarchical NN search and a separate level-local rewire pass with an unsound bound. The rest (outlink
and inlinks, then repair of the affected clusters only) is the same shape. Deltas D3, D4, D5.

---------------------------------------------------------------------------------------------------
## 3. Deletion

- **No public deletion of a level-0 element.** `delList` starts empty and only `addList` is filled at
  level 0 (VC:50, VC:56-63). No service or subscriber removes a view (VM:140-151). The deletion code
  runs only at levels ≥ 1, after `prepareUpperLevel` fills `delList` with the parent elements of demoted
  representatives (VC:208, VU:1294-1311).
- **What the deletion code does** (VC:117-205), for each `del_elem`:
  - Each inlinker re-finds its NN with `del_elem` excluded: VC:154
    `this->getNNElement(elements_at_level, *inlink_elem, dist, del_elem);`. The exclusion is done by
    asking for k=2 and skipping the first hit if it is `del_elem` (VC:291-294, VC:309-317). If no other
    element exists, the inlinker's outlink is cleared and its cluster is re-elected (VC:156-172).
    Otherwise `updateLinks` and `updateCluster` run (VC:178, VC:184), which re-runs the component split
    and merge.
  - **No rewire sweep** follows a deletion. This matches SPEC:283-287.
  - `delOutLink` drops `del_elem → nn` (VC:409-435). `updateNNClusterDelElem` (VC:901-948) requires that
    `del_elem` has no inlinks left (VC:908-911). It removes `del_elem` from the NN's cluster
    (`delChildElement`, VC:931) and deletes the cluster if it is empty (VC:933-938). Otherwise it
    re-elects and recomputes the radius (VC:942).
  - `delElement` erases the element from the level's map (VC:204).
- **Representative role upward.** The ROS code does not delete an element "from all levels". When a
  level-l cluster loses its representative, `getAddDelList` puts the old representative's level-(l+1)
  copy into `delList` (VU:1296-1310, via the child cluster's `parent_key`) and the new representative's
  copy into `addList` (VU:1327-1348). The next iteration of the main loop processes both. The process
  cascades for as long as representatives keep changing.
  Risk: VU:1307-1309 dereferences `getElement(level+1, old_cit->second.parent_key)` without a NULL check.
- Many older deletion variants are commented out (VC:678-1087). They are dead.

**Spec comparison.** The spec's `unlink!` (SPEC:281-292) has the same structure as the ROS upper-level
deletion: re-link the inlinkers with the deleted element excluded, no rewire, then repair. The spec
exposes it at level 0 as `delete!` (SPEC:105, :305). ROS never deletes a view, so delete as a
user-facing operation is new relative to 2015. It does exist in the C++ (BRIEF:280-303). Delta D13.

---------------------------------------------------------------------------------------------------
## 4. Representative election

`ViewClusteringUtils::selectRepresentative` (VU:2450-2572), called through
`updateClusterRepresentiveAndRadius` (VU:2574-2580). Exact rule as coded:

```
VU:2478  if (it->second.inlink_keys.size() > num_child)
VU:2480      num_child = it->second.inlink_keys.size(); candidates.clear(); candidates.push_back(it->first);
VU:2483  } else if (it->second.inlink_keys.size() == num_child) { candidates.push_back(it->first); }
...
VU:2505                  sum += inlink_em->second.outlink_distance;
VU:2509              if (sum < min)
VU:2511                  min = sum; rep = *cand_it;
```

The order of criteria is:
1. **Most inlinks** at this level.
2. **Smallest sum of the inlinkers' `outlink_distance`**, in the raw units of the distance (§1), with no
   unit conversion.
3. **First in `member_keys` order** (strict `<`). `member_keys` order depends on the path:
   - lexicographic key order after `getAllLinks` (VU:628);
   - append order after `addElementintoNNCluster` (VC:456) or `mergeClusters` (VU:788-789).

`rep` is initialised to the current representative (VU:2453), so it survives only if no candidate's sum is
below `FLT_MAX`. A representative change sets `c_status = UPD` unless the cluster is `ADD` (VU:2525-2529).
The cluster key is not renamed (`clust_key_change = false`, VU:2533).

**Spec comparison.** SPEC:220-222 uses most inlinks, then the smallest sum of the inlinkers' outlink
distances (metric units), then the smallest id. Criteria (1) and (2) match ROS. For (2), "metric units"
differs from ROS only where the spec converts units (SqEuclidean is handled in Euclidean units,
SPEC:136-140). For a `UserMetric` it is the same as ROS. Criterion (3) differs: ROS uses member order,
the spec uses the smallest id. ROS and the C++ agree with each other (BRIEF:208-215: raw units, then
position in the member list). SPEC §12 already documents the C++ divergence (SPEC:504). Delta D6.

---------------------------------------------------------------------------------------------------
## 5. Cluster radius

`ViewClusteringUtils::updateClusterRadius` (VU:1908-1955):

```
VU:1931  this->getDistBetweenViews(it->second.view_key, elem->view_key, dist);
VU:1935  PerceptualCluster* child_cluster = this->getChildClusterwithElem(it->second.level, member_it->c_str());
VU:1936  if (child_cluster==NULL && elem->level!=0) { //child cluster must be deleted.
VU:1938      continue;
VU:1939  } else if (child_cluster==NULL) { child_radius = 0;
VU:1943  } else { child_radius = child_cluster->radius; }
VU:1950  if (radius<dist+child_radius) radius = dist+child_radius;
```

- **Definition.** `radius(C) = max over m in C of ( d(m, rep) + radius(child(m)) )`. Here `child(m)` is the
  level-(l−1) cluster whose `parent_key == m` (VU:1018-1026). There is a fallback that matches the
  child's representative by `view_key` and rewrites its `parent_key` (VU:1028-1048). The child term is 0
  at level 0. This is the **recursive** definition. It is not "farthest member" and not an argmax over
  inlinks.
- **Ambiguity.** The message comment says otherwise. MSG/PerceptualCluster:14:
  `float32 radius ## The cluster radius, which is the distance from the representative to the farthest neighbor in the cluster`.
  The code is recursive. The comment at VU:1906 `// function not yet called` is stale, because the
  function is called from VU:2577.
- **Under-estimate.** A member at level > 0 with no findable child cluster is **skipped** (VU:1936-1938).
  The resulting radius is too small, which is unsafe for pruning.
- **When it is refreshed.** Only when `updateClusterRepresentiveAndRadius` is called on that very cluster.
  The call sites are all at the cluster's own level, after a membership or link change: VC:169, :457,
  :462, :468, :517, :578, :617, :664, :942 and VU:467. **Ancestors are not refreshed.** Suppose a
  level-0 cluster grows but keeps its representative. Then `getAddDelList` produces no add or del for
  level 1 (VU:1294-1349 compare only representative keys), nothing is done at level 1, and the parent
  cluster's radius stays stale (too small). The CHN/`check_mod` path (VU:2146-2164) widens the search
  only for clusters whose status is not NOC. That status is reset per level (VC:1127 →
  `resetCStatus`), so it does not cover this case. The end-of-insert `resetAllCStatus` is a no-op
  because its loop condition is inverted (VU:1125 `for (uint32_t i=0; _clusters.size()<i ; i++)`).
- **Units.** Raw units of the view distance (squared-L2 MHD, §1). There is no metric conversion.

**Spec comparison.** SPEC:239-240 adds a middle term `M(m, rep(child(m)))`, which is 0 at rest. With that
term the spec's definition equals the ROS one at rest. The spec refreshes ancestors along the root paths
(SPEC:277-278). ROS does not. The spec's `maxout` has no ROS counterpart. The C++ also refreshes along the
seed paths (BRIEF:244-246) and uses metric units (BRIEF:183-192). Delta D7, D8.

---------------------------------------------------------------------------------------------------
## 6. Upper levels

- **Population.** Level l+1 holds copies of the level-l representatives (`newParentElement`,
  VU:362-371). The child cluster's `parent_key` is set to the copy's key (VU:1323, VU:1346).
- **Incremental, not rebuilt.** Each level receives an `addList` and a `delList` computed by
  `prepareUpperLevel` → `getAddDelList` (VC:1090-1141, VU:1147-1357). The live branch is the
  `else // enumerate to check two lists` branch, because `bool _incremental = false;` (VU:1165).
  - `delList` = old clusters whose representative key is absent from the new clusters (VU:1294-1311).
  - `addList` = all representatives if the level previously had one cluster (VU:1312-1324). Otherwise it
    is the new representatives absent from the old clusters (VU:1327-1348).
  - These are net set differences of the representative sets, which is the spec's "net composition"
    (SPEC:317-320).
  - At level l+1, the same insert routine (hierarchical NN, link, join cluster, level-local rewire) and
    the same delete routine are applied (VC:74-205).
  - Within a level, adds are processed **before** deletes (VC:74, VC:117). The spec does unlink before
    link (SPEC:312-313).
  - The from-scratch builder `rebuildCluster` (VU:1055-1112) has no live caller; it is commented out at
    VC:214-217.
- **New top level.** `getClustersAtLevel` and `getElementsAtLevel` append an empty level whenever
  `level == size()` (VU:46-50, VU:62-66). The loop climbs while the current level has more than one
  cluster (VC:228 `} while (clusters_at_level->size()>1);`).
- **Dropping levels.** VC:1121-1124: when a level collapses to one cluster but had more before, the code
  calls `clearUpperLevel`. That function erases **only one** level index (VU:1136-1142
  `_clusters.erase(_clusters.begin() + upperLevel); ... _elements.erase(_elements.begin() + upperLevel);`),
  and only if that level is non-empty. If two or more levels existed above, the higher ones shift down
  and remain stale. The root is taken as `_elements.size()-1` (VU:84), so a stale level would become
  the root. This is a likely bug. SPEC:309 drops every level ≥ l.
- **Order independence: no.** Ties are broken by lexicographic map order and by unstable `std::sort` on
  distance only (VU:2050, VUH:68-71). A rewire requires strict `<` and is skipped at distance 0
  (VC:561). Representative ties fall to member order (§4). The rewire bound is unsound (§2), radii can
  be stale (§5), and the distance is not a metric (§1), so the stored NNG can depend on insertion
  order. Views within a teaching event are inserted in `getRTOVs` order (VC:53).

Other defects on this path:
- VC:1115 `clusters->erase(c_it);` does not advance the iterator, which is undefined behaviour.
- `getClusterNewKey` never increments its retry counter `it` and does `return NULL;` for a `std::string`
  (VU:280-291).

**Spec comparison.** The spec's `propagate` (SPEC:307-315) is the same design as ROS: a per-level delta of
representatives, then the same link and unlink at every level. **ROS is closer to the spec than the C++
is**: the C++ brute-forces over S at each upper level and re-clusters each dirty level in full
(BRIEF:250-268). ROS uses hierarchical NN and per-cluster repair at every level. The spec claims order
independence (SPEC:227-230). ROS does not have it. Deltas D9-D12.

---------------------------------------------------------------------------------------------------
## 7. Search and recognition

- **Services.** VM:140 `RecognizeSITOVs` (request `race_perception_msgs/SITOV[] sitovs`, response
  `string object_name`, SRV/RecognizeSITOVs:1-3). VM:142 `exhRecognizeinDB` and VM:145 `phRecognizeinDB`
  take an `rtov_key` (SRV/RecognizeSITOVsinDB:1-4). `cb_RecognizeSITOVsinDB` (VM:227-242), which runs
  both and compares them, is defined but **never advertised**. SIM:148-159 calls ph and exh separately.
- **The query is one view.** It is a variable-size set of spin images (`vector<SITOV>`), matched against
  stored views. The answer is the `object_name` of the **single nearest view**:
  VC:248-250 `PerceptualElement* nn = _vcUtils->getNNElementwithPH(0, src_view, dist); if (nn) return nn->object_name;`.
  There is no k-NN, no voting across views, and no query made of several views.
- **Descent.** `getNNElementwithPH(level, src_view, dist)` (VU:2221-2279) is level-synchronous from the
  root members. At each level it computes `d` to the whole current list and takes `min` and `index`
  (VU:2236-2246). Then, for each element, it keeps the child cluster's members according to the
  predicate below.
- **Sign bug, confirmed.**
  ```
  VU:2259  //if (i == index || child_cluster->radius+r.at(i) <= min)
  VU:2260  if (i == index || r.at(i)-child_cluster->radius >= min)      // keeps clusters whose LOWER BOUND >= min: flipped
  VU:2338  if (nn_clst == &it->second || dists.at(i)-it->second.radius < min) {   // getNNClusterswithRadius: correct keep rule
  ```
  The correct rule keeps C iff `d − r < min`, so it prunes iff `d − r ≥ min` (non-strict). The
  recognition descent keeps the nearest cluster plus every *far* cluster and drops the near ones. The
  result is approximate, and wrong in the direction that matters. The insertion search (§2) uses the
  correct direction.
- **More recognition defects.**
  - The returned distance is stale. `dist = min` is assigned (VU:2272) only after the break test at
    VU:2249-2250, so the caller gets the previous level's minimum.
  - The function returns a pointer into a local vector, `return &peList.at(index);` (VU:2278). The
    caller then dereferences a dangling pointer (VC:249-250).
  - If `getRootCluster()` returns NULL (empty index), it is dereferenced at VU:2224.
- **Exhaustive alternative.** `getExhaustiveNNObject` (VC:255-277) is a full scan with strict `<`.
- **k-NN inside ROS.** `getkNNElementwithPH` accepts `k` (VUH:297 `const uint32_t k=1`) and is used with
  k=1 and k=2 (VC:290-294). So a k-NN search existed, but only internally for insert and delete.

**Spec comparison.** The spec gives an exact k-NN with best-first search (SPEC:331-344). The level-sync
oracle "of the 2015 algorithm (prune with the same strict rule)" (SPEC:346-348) is the 2015 *shape* with
the corrected, strict predicate. The spec has no label output and no multi-view query. Deltas D14-D16.

---------------------------------------------------------------------------------------------------
## 8. Navigation and the perceptual hierarchy as output

The following are readable without computing any distance:
- **Per element:** `outlink_key`, `outlink_distance`, `inlink_keys`, `object_name`, `view_key`, `level`
  (MSG/PerceptualElement:9-15).
- **Per cluster:** `level`, `representative_key`, `parent_key` (the key of the level+1 *element*, not a
  cluster: MSG/PerceptualCluster:12 `string parent_key ## The parent element key in upper level (ELEM)`),
  `member_keys`, `radius`, `c_status` (MSG/PerceptualCluster:9-15).
- **Accessors:**
  - `getRootCluster` (VU:72-80)
  - `getChildClusters` (VU:158-175) and `getChildClusterwithElem` (VU:1005-1053)
  - `getMemberElements` (VU:136-156)
  - `getClusterwithElement` and `getClusterwithRep` (VU:797-866)
  - `getAllLinks`, the connected component (VU:602-640)
  - **`getLeafElementswithCluster`**, all level-0 elements under a cluster (VU:896-921)
  - `getLeafRepElementswithCluster` (VU:923-957)
- **Hierarchy text output.** This runs on every teaching event. `toETEHierarchy` (VU:1711-1782) builds a
  Newick-like string for the ETE Python library, with the cluster key and radius per internal node and
  `leaf-label` per leaf (VU:1731, :1754, :1772). It is published on `/perception/perceptual_hierarchy/pub_ph`
  (VM:135, VM:188-193). The published string also has the object-representation summary appended
  (VM:190-191).
- **ObjectRepresentation.** `buildObjectRepresentations` (VU:1439-1458) and `addElementsinToObjRep`
  (VU:2374-2431) walk down from the root. For every maximal subtree whose leaves all carry one
  `object_name` (`sameLabel`, VU:2347-2372), they record `(rep key, cluster radius)` under that object.
  Heterogeneous bottom clusters contribute each member with radius 0 (VU:2421-2427). The message is
  MSG/ObjectRepresentation:8-11 (`elem_keys`, `elem_radiuses`). Nothing in the package reads it back. It
  is not used by recognition (VC:245-253).
  - Defect: `matchObjectList` overwrites `exists` on every iteration (VU:1497).
- **dot file.** `drawHierarchy`/`drawLevel`/`drawCluster` (VU:1562-1669) have **no caller**. The `.dot`
  write is under `if (false)` (VM:196-199). The `dot_file` parameter only decides whether a GTK render
  thread starts (VM:166-167).
- **Printing.** `printHierarchy` (VU:1784-1837) and `printClusterMap` (VU:1858-1880). The latter is a
  partial invariant check: it compares the element count with the summed cluster membership (VU:1878-1879).

**Missing from spec §7** (SPEC:366-385):
- (a) all level-0 leaves under a cluster (ROS `getLeafElementswithCluster`). This can be built from
  `children`, but it is the ROS's most-used traversal (VU:892, :1446, :2389).
- (b) a tree export with radii and labels (ROS `toETEHierarchy`).
- (c) the label-pure subtree decomposition (ObjectRepresentation).

Item (c) needs labels, which the spec does not store. The spec's `parent(idx, c)` returns a cluster,
while ROS `parent_key` names an element. The two are equivalent, because the spec keeps one id across
levels. Deltas D17-D19.

---------------------------------------------------------------------------------------------------
## 9. Parameters

- PARAMS:2-3 `verbose: true`, `dot_file: true`. Read at VM:129-130. Neither one affects the structure.
- Hard-coded switches in code:
  - `bool _bknn = true;` (VC:283): hierarchical vs exhaustive NN at insert.
  - `_inc = false` (VC:27): whether cleared clusters are physically deleted (VC:632-633, :924-925, :937-938).
  - `bool _incremental = false;` (VU:1165): the alternative add/del-list computation.
  - `_cat_level = true` (VU:27): ObjectRepresentation groups by category, stripping the name after the
    last `_` (VU:1369-1373, :1502-1507).
  - `bool clust_key_change = false;` (VU:534, VU:2533).
  - `max_it = 100` (VU:281).
  - `k` = 1 or 2 (VC:290-294).
  - `check_mod = true` (VC:300).
- Feature-side constants: 153 values per spin image (SITOV:12) and the 1e6 cap (SI:281). The spin-image
  extraction parameters live in `race_feature_extraction`, outside this package.
- **The structure itself is parameter-free.** No cluster size, threshold, M, ef or level count appears
  in the algorithm. This matches SPEC:31-33.

---------------------------------------------------------------------------------------------------
## 10. Other ROS behaviour the spec does not cover

- **Labels on elements.** `object_name` is stored on every element (MSG/PerceptualElement:15, VU:356),
  copied to parents (VU:367), and returned by recognition (VC:250). The spec has no labels.
- **Category vs instance naming.** With `_cat_level`, the name is truncated at the last `_`
  (VU:1369-1373).
- **Cluster status flags.** `c_status` NOC/ADD/DEL/UPD (VUH:23-29) marks clusters changed during an
  update. The insert search uses them (VU:2041-2044, :2154-2155), and the drawing and printing code
  skips DEL. They are the ROS counterpart of the spec's "floating cluster" reachability contract
  (SPEC:294-299). They are not persisted, and the end-of-insert reset is broken (VU:1125).
- **Search event types.** `d_type` DEI/DNN/DMR/DPR/CHN (VUH:31-38) belong to the k-count sweep (§2).
- **Statistics.** `getObjectRepresentation` returns `"#objects:%lu, objReps:%lu, elements:%lu"` (VU:1470).
  No distance counters exist.
- **Purity and evolution logging.** The label-pure decomposition (§8) and the Newick hierarchy are
  published after every teaching event (VM:188-193). Together they form an evolution log of the
  hierarchy. Recognition agreement between ph and exh was measured externally (SIM:148-159).
- **Multi-feature keys.** These were planned but not implemented. MSG/PerceptualElement:5
  `### To support multiple features ==> Key: ELEM_<feature_type>_<level>_<element_id>` and :10
  `#string ft_type` are commented out. The same holds for MSG/PerceptualCluster:5, :10.
- **Persistence.** None. The structure lives in `_clusters` and `_elements` in memory (VUH:82-83). The
  messages document DB key formats (MSG/*:4) but nothing writes them. `_pdb` is used only to read
  views and categories and to build keys (every `_pdb->` call is at VC:53, VU:179-185, :274-285, :1391,
  :1520-1540, :1973-1985, :2283-2317). Spec save/load (SPEC:118) is new.
- **Batch teaching per object.** All views of a track are inserted under one label in one call (VC:53-63).
  The spec's `add!(idx, objs)` (SPEC:104) covers the mechanics but not the label.
- **Oracles in the code.** `getExhaustiveNNObject` (VC:255-277), `getNNElementExhs` (VC:321-370) and
  `updateTheOtherElementsExhs` (VC:474-524) are brute-force twins of the hierarchical routines. The spec's
  oracle tests (SPEC:419-428) play the same role.

---------------------------------------------------------------------------------------------------
## 11. Spec deltas

| # | Topic | ROS behaviour | Spec behaviour | Recommendation |
|---|---|---|---|---|
| D1 | Element distance used in 2015 | View = variable set of 153-d spin images. `max(mean_i min_j ‖a_i−b_j‖², mean_j min_i ‖·‖²)` with a 1e6 cap (SI:252-312, VU:1957-1969). Not a metric | Generic `UserMetric(is_metric=false)`; tests a "mean-closest-descriptor" non-metric (SPEC:144-151, :422-424) | amend spec: in §10 `objects.jl`, define the non-metric test distance as exactly the ROS formula (max-symmetrised, squared L2 per descriptor), cite SI:261-316, and state that the 2015 system pruned with it |
| D2 | Pruning under a non-metric | Prunes anyway with radii, so exactness is not guaranteed (VU:2040-2046, :2338) | Non-metric runs unpruned and stays exact (SPEC:148-151) | keep spec; ask author: should an opt-in "prune anyway (heuristic, ROS-compatible)" mode exist for the robot use case, or is unpruned-exact the only behaviour? |
| D3 | NN search at insert | Hierarchical descent with a k-count sweep over (d−r, d, d+r) events; changed clusters added unconditionally (VU:1989-2219, VC:283-300) | One pruned best-first descent with LB and memo (SPEC:257-270) | keep spec |
| D4 | Rewire candidate bound | Level-local: keep C iff `d−r < d(e, nearest rep)` (VU:2338). Unsound, can miss rewires. Same as the C++ before fe7641f (BRIEF:174-176) | Skip only if `LB > best` **and** `LB > maxout[l](C)` (SPEC:266-268) | keep spec; amend spec §12: add a row noting the ROS bound is the pre-fe7641f one |
| D5 | Rewire to an exact duplicate | Never rewires at distance 0: `dist != 0.0 &&` (VC:561) | Lexicographic `(M, id)` compare; a duplicate at distance 0 rewires | keep spec (needed for an exact 1-NN graph) |
| D6 | Representative tiebreak | Max inlinks → min raw sum of inlinkers' outlink distances → first in `member_keys` order (lexicographic key order or append order) (VU:2478-2513) | Same first two keys in metric units, then smallest id (SPEC:220-222) | keep spec; amend spec §12 row "Representative tiebreak" to say ROS = C++ here (raw units, member order) |
| D7 | Radius definition | Recursive `max(d(m,rep) + radius(child(m)))`; a member is skipped if its child cluster is missing (VU:1931-1950). The message comment says "farthest neighbour" (MSG/PerceptualCluster:14) | Recursive plus middle term; equal at rest (SPEC:239-240) | keep spec |
| D8 | Radius maintenance | Refreshed only on the touched cluster; ancestors stay stale when a representative is unchanged (§5) | Refresh along root paths (SPEC:277-278); `maxout` per level | keep spec |
| D9 | Upper-level update | Incremental per-level add/del of representatives with the same insert/delete routines (VC:66-228, VU:1294-1348) | `propagate(delta)` with `link!`/`unlink!` (SPEC:307-315) | keep spec; amend spec §12: note that the path-local upper-level design follows the 2015 ROS, unlike the C++ brute force plus re-cluster (BRIEF:250-268) |
| D10 | Within-level operation order | Adds then deletes (VC:74, :117) | Unlink then link (SPEC:312-313) | keep spec (canonical at rest) |
| D11 | Level drop | Erases one level only (VU:1136-1142), a likely bug | Drop every level ≥ l (SPEC:309) | keep spec |
| D12 | Order independence | Not order-independent: map-key and unstable-sort ties, unsound rewire, stale radii | Claimed and tested (SPEC:227-230, :429-431) | keep spec; amend spec §1 or §12 to state that order independence is new relative to 2015 (do not attribute it to the ROS system) |
| D13 | Delete | No user-facing delete of a view; deletion code only demotes upper-level representatives (VC:50, :117-205) | `delete!` first-class (SPEC:105, :305) | keep spec; ask author: the manuscript's "online delete" claim should cite the C++ (`markDelete`), not the 2015 ROS. Agree? |
| D14 | Query form and answer | One view → `object_name` of the 1-NN view (VC:245-253); no k, no multi-view voting | `knn(idx, q, k)` → ids and distances (SPEC:107) | ask author: add an optional label vector plus a `recognize(idx, q)` = label of the 1-NN, or leave labels to the caller keyed by id? |
| D15 | Recognition prune sign | Flipped: keep iff `d−r >= min` (VU:2260); the correct rule is in `getNNClusterswithRadius` (VU:2338) | Strict correct rule in both engines (SPEC:337-339, :346-348) | keep spec; amend spec §6.5: replace "the same strict rule" with "the corrected rule; ROS `getNNElementwithPH` has it flipped (VU:2260)" |
| D16 | Level-sync predicate strictness | Correct ROS rule prunes iff `d−r ≥ min` (non-strict) (VU:2338), as the C++ does (BRIEF:329) | Strict `>` with guard (SPEC:507) | keep spec; amend spec §12 row "Level-sync prune" to add "ROS: `>=` as well" |
| D17 | Leaves under a cluster | `getLeafElementswithCluster` (VU:896-921), heavily used | Only `members`/`children` (SPEC:372-374) | amend spec §7: add `leaves(idx, c)` (level-0 ids under c, no distance evaluations) |
| D18 | Hierarchy export | Newick (ETE) string with radii and leaf labels, published each teach (VU:1711-1782, VM:135, :188-193); the dot export is dead (VM:196) | Graphs.jl `level_graph` only (SPEC:379-385) | ask author: add `newick(idx; labels=nothing)` for figures and evolution logging? |
| D19 | Label-pure decomposition | ObjectRepresentation of maximal single-label subtrees with radii (VU:2374-2431) | None | ask author: out of scope, a README example, or a navigation utility taking an external label vector? |
| D20 | Cluster status flags | NOC/ADD/DEL/UPD guide the in-update search (VUH:23-29, VU:2146-2164) | Reachability contract, floating clusters (SPEC:294-299) | keep spec |
| D21 | Persistence | None; in-memory only (VUH:82-83) | `save`/`load` (SPEC:118) | keep spec |
| D22 | Parameters | Structure parameter-free; only `verbose`, `dot_file` and hard-coded switches (PARAMS:2-3, VC:283) | Parameter-free (SPEC:31-33) | keep spec |
| D23 | Multi-feature keys | Planned in message comments, not implemented (MSG/PerceptualElement:5, :10) | Not covered | keep spec (out of scope; a composite `UserMetric` covers it) |

**Delta count: 23.**

By leading recommendation:
- 18 keep spec. Six of these also carry a wording amendment (D4, D6, D9, D12, D15, D16). Two carry a
  question for the author (D2, D13).
- 2 amend spec (D1, D17).
- 3 ask author (D14, D18, D19).

### Where ROS and the C++ brief disagree

- **Upper levels.** ROS: hierarchical NN plus per-cluster repair at every level. C++: brute force over S
  plus a full re-cluster of dirty levels (BRIEF:250-268).
- **Radius refresh.** ROS: none upward. C++: along the seed paths (BRIEF:244-246).
- **Radius units.** ROS: raw. C++: metric via `to_metric` (BRIEF:183-192).
- **Rewire bound.** ROS: the pre-fe7641f form. C++: with `max_outlink_m` (BRIEF:165-176).
- **Level drop.** ROS: one level. C++: `clearLevelsAbove_` clears all (BRIEF:269-271).
- **Delete.** ROS: upper-level demotion only. C++: `markDelete` at level 0 (BRIEF:280-303).
- **Search.** ROS: 1-NN with the flipped sign. C++: exact k-NN.

ROS and the C++ **agree** on the representative rule (BRIEF:194-215) and on the strict `<` for rewiring.
