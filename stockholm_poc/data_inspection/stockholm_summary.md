# OSM Summary Report

- Source: `Stockholm.osm.pbf`
- Driver: `OSM`
- Generated: `2026-04-19T13:28:36.119415+00:00`
- Layer count: `5`
- Layer names: `points`, `lines`, `multilinestrings`, `multipolygons`, `other_relations`
- Total features: `951,811`

## Feature Class Counts

- `area_like_features`: 450,189
- `edge_like_features`: 270,987
- `line_features`: 270,987
- `multipolygon_features`: 450,189
- `node_like_features`: 224,908
- `point_features`: 224,908

## Derived Theme Counts

- `building_like_features`: 369,782
- `poi_like_features`: 87,430
- `road_like_features`: 228,902

## Extra GDAL Layers

- `multilinestrings`: geometry `Multi Line String`, features `2,457`
- `other_relations`: geometry `Geometry Collection`, features `3,270`

## Layer: `points`

- Geometry type: `Point`
- Feature count: `224,908`
- Derived counts: `building_like_features`=0, `poi_like_features`=50,411, `road_like_features`=0
- Fields: `osm_id`, `name`, `barrier`, `highway`, `ref`, `address`, `is_in`, `place`, `man_made`, `other_tags`
- Top non-null fields:
  - `osm_id`: 224,908
  - `other_tags`: 199,853
  - `highway`: 58,622
  - `name`: 34,584
  - `barrier`: 9,115
  - `place`: 4,761
  - `ref`: 4,074
  - `man_made`: 1,505
  - `address`: 2
- Top `other_tags` keys:
  - `addr:housenumber`: 74,541
  - `addr:street`: 74,415
  - `addr:country`: 68,111
  - `addr:city`: 54,312
  - `entrance`: 44,880
  - `addr:postcode`: 35,024
  - `amenity`: 28,188
  - `support`: 19,725
  - `lamp_type`: 19,612
  - `lamp_mount`: 19,008
  - `natural`: 12,881
  - `check_date`: 11,465
  - `crossing`: 11,210
  - `tactile_paving`: 9,193
  - `opening_hours`: 8,351
- Sample features:
  - Sample 1: `fid`=119024, geometry=`POINT`, property keys=highway, osm_id
  - Sample 2: `fid`=119030, geometry=`POINT`, property keys=highway, osm_id, other_tags
  - Sample 3: `fid`=119038, geometry=`POINT`, property keys=highway, osm_id, other_tags

## Layer: `lines`

- Geometry type: `Line String`
- Feature count: `270,987`
- Derived counts: `building_like_features`=0, `poi_like_features`=275, `road_like_features`=228,902
- Fields: `osm_id`, `name`, `highway`, `waterway`, `aerialway`, `barrier`, `man_made`, `railway`, `z_order`, `other_tags`
- Top non-null fields:
  - `osm_id`: 270,987
  - `z_order`: 270,987
  - `highway`: 228,902
  - `other_tags`: 185,311
  - `name`: 64,787
  - `barrier`: 14,241
  - `man_made`: 7,278
  - `waterway`: 4,863
  - `railway`: 4,403
  - `aerialway`: 25
- Top `other_tags` keys:
  - `surface`: 110,204
  - `maxspeed`: 50,891
  - `lit`: 34,097
  - `foot`: 32,337
  - `bicycle`: 26,436
  - `service`: 22,892
  - `oneway`: 20,486
  - `lanes`: 12,688
  - `segregated`: 10,051
  - `layer`: 9,098
  - `footway`: 9,019
  - `smoothness`: 7,572
  - `access`: 7,518
  - `sidewalk`: 6,548
  - `wikidata`: 6,111
- Sample features:
  - Sample 1: `fid`=1240, geometry=`LINESTRING`, property keys=highway, name, osm_id, other_tags, z_order
  - Sample 2: `fid`=1241, geometry=`LINESTRING`, property keys=highway, name, osm_id, other_tags, z_order
  - Sample 3: `fid`=1242, geometry=`LINESTRING`, property keys=highway, name, osm_id, other_tags, z_order

## Layer: `multilinestrings`

- Geometry type: `Multi Line String`
- Feature count: `2,457`
- Derived counts: `building_like_features`=0, `poi_like_features`=2, `road_like_features`=0
- Fields: `osm_id`, `name`, `type`, `other_tags`
- Top non-null fields:
  - `osm_id`: 2,457
  - `type`: 2,457
  - `other_tags`: 2,455
  - `name`: 2,427
- Top `other_tags` keys:
  - `route`: 2,449
  - `network`: 2,237
  - `ref`: 1,909
  - `from`: 1,748
  - `to`: 1,735
  - `public_transport:version`: 1,643
  - `network:wikidata`: 1,565
  - `colour`: 811
  - `operator`: 427
  - `network:wikipedia`: 363
  - `osmc:symbol`: 289
  - `via`: 247
  - `distance`: 242
  - `roundtrip`: 189
  - `wikidata`: 128
- Sample features:
  - Sample 1: `fid`=29233, geometry=`MULTILINESTRING`, property keys=name, osm_id, other_tags, type
  - Sample 2: `fid`=29234, geometry=`MULTILINESTRING`, property keys=name, osm_id, other_tags, type
  - Sample 3: `fid`=31382, geometry=`MULTILINESTRING`, property keys=name, osm_id, other_tags, type

## Layer: `multipolygons`

- Geometry type: `Multi Polygon`
- Feature count: `450,189`
- Derived counts: `building_like_features`=369,782, `poi_like_features`=36,720, `road_like_features`=0
- Fields: `osm_id`, `osm_way_id`, `name`, `type`, `aeroway`, `amenity`, `admin_level`, `barrier`, `boundary`, `building`, `craft`, `geological`, `historic`, `land_area`, `landuse`, `leisure`, `man_made`, `military`, `natural`, `office`, `place`, `shop`, `sport`, `tourism`, `other_tags`
- Top non-null fields:
  - `osm_way_id`: 446,384
  - `building`: 369,782
  - `other_tags`: 180,551
  - `landuse`: 33,099
  - `amenity`: 18,161
  - `name`: 16,662
  - `natural`: 15,954
  - `leisure`: 12,644
  - `place`: 4,564
  - `type`: 3,829
  - `osm_id`: 3,805
  - `sport`: 3,217
  - `man_made`: 772
  - `shop`: 604
  - `barrier`: 540
- Top `other_tags` keys:
  - `addr:street`: 121,701
  - `addr:housenumber`: 121,321
  - `addr:country`: 112,348
  - `addr:city`: 90,002
  - `addr:postcode`: 85,498
  - `building:levels`: 44,160
  - `roof:shape`: 20,950
  - `roof:levels`: 11,821
  - `roof:colour`: 11,572
  - `access`: 8,283
  - `building:colour`: 6,789
  - `start_date`: 6,229
  - `parking`: 5,754
  - `building:material`: 5,098
  - `roof:material`: 3,792
- Sample features:
  - Sample 1: `fid`=2380, geometry=`MULTIPOLYGON`, property keys=landuse, osm_id, type
  - Sample 2: `fid`=3653, geometry=`MULTIPOLYGON`, property keys=name, natural, osm_id, other_tags, type
  - Sample 3: `fid`=8043, geometry=`MULTIPOLYGON`, property keys=landuse, osm_id, type

## Layer: `other_relations`

- Geometry type: `Geometry Collection`
- Feature count: `3,270`
- Derived counts: `building_like_features`=0, `poi_like_features`=22, `road_like_features`=0
- Fields: `osm_id`, `name`, `type`, `other_tags`
- Top non-null fields:
  - `osm_id`: 3,270
  - `type`: 3,268
  - `other_tags`: 1,276
  - `name`: 377
- Top `other_tags` keys:
  - `restriction`: 544
  - `public_transport`: 267
  - `colour:back`: 164
  - `colour:text`: 164
  - `colour:arrow`: 163
  - `destination`: 162
  - `enforcement`: 132
  - `check_date`: 95
  - `destination:ref`: 76
  - `destination:symbol`: 48
  - `wikidata`: 46
  - `network`: 45
  - `ref`: 37
  - `netex:stopplace:id:SE-SL`: 33
  - `except`: 31
- Sample features:
  - Sample 1: `fid`=8323, geometry=`GEOMETRYCOLLECTION`, property keys=name, osm_id, other_tags, type
  - Sample 2: `fid`=8324, geometry=`GEOMETRYCOLLECTION`, property keys=name, osm_id, other_tags, type
  - Sample 3: `fid`=68014, geometry=`GEOMETRYCOLLECTION`, property keys=name, osm_id, other_tags, type
