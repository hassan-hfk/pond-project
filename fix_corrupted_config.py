"""
Fix corrupted config.yaml file
Restores ROIs, Calibration, and classes_trigger to proper format
"""

import yaml
from pathlib import Path

# Define config path
project_root = Path(__file__).parent
CONFIG_PATH = project_root / 'config' / 'config.yaml'

print("🔧 Config Fixer Tool")
print("=" * 50)

# Load current config
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

print(f"\n📋 Current config sections: {list(cfg.keys())}")

fixed = False

# Fix ROIs if corrupted
if 'rois' in cfg:
    rois = cfg['rois']
    print(f"\n🔍 Checking ROIs...")
    print(f"   Type: {type(rois)}")
   
    if isinstance(rois, dict):
        print("   ❌ ROIs corrupted (dict instead of list)")
        print(f"   Corrupted data: {rois}")
        cfg['rois'] = []
        print("   ⚠️  Set to empty - please use ROI Editor to redraw")
        fixed = True
    elif isinstance(rois, list):
        print("   ✅ ROIs format is correct (list)")
        if len(rois) > 0:
            print(f"   📍 {len(rois)} ROI polygon(s) found")
    else:
        print(f"   ❌ Unexpected ROI type: {type(rois)}")
        cfg['rois'] = []
        fixed = True

# Fix Calibration if corrupted
if 'calibration' in cfg:
    cal = cfg['calibration']
    print(f"\n🔍 Checking Calibration...")
    print(f"   Type: {type(cal)}")
   
    if isinstance(cal, dict):
        # Check if it has the required fields
        required_fields = ['focal_px', 'ref_height_m', 'ref_distance_m']
        missing = [f for f in required_fields if f not in cal]
       
        if missing:
            print(f"   ⚠️  Missing fields: {missing}")
        else:
            print(f"   ✅ Calibration data present:")
            print(f"      Focal length: {cal.get('focal_px')} px")
            print(f"      Ref height: {cal.get('ref_height_m')} m")
            print(f"      Ref distance: {cal.get('ref_distance_m')} m")
            print(f"      VP: {cal.get('vertical_vp')}")
       
        # Check for corrupted nested structures
        if 'ref_box_norm' in cal:
            ref_box = cal['ref_box_norm']
            if isinstance(ref_box, dict):
                print("   ❌ ref_box_norm corrupted (dict instead of list)")
                print(f"   Corrupted data: {ref_box}")
                # Can't reliably fix, remove it
                del cal['ref_box_norm']
                print("   ⚠️  Removed corrupted ref_box_norm - please recalibrate")
                fixed = True
            elif isinstance(ref_box, list) and len(ref_box) == 4:
                print(f"   ✅ ref_box_norm format correct: {ref_box}")
       
        if 'vertical_vp' in cal:
            vp = cal['vertical_vp']
            if vp is not None:
                if isinstance(vp, dict):
                    print("   ❌ vertical_vp corrupted (dict instead of list)")
                    cal['vertical_vp'] = None
                    print("   ⚠️  Set vertical_vp to null - please recalibrate")
                    fixed = True
                elif isinstance(vp, list) and len(vp) == 2:
                    print(f"   ✅ vertical_vp format correct: {vp}")
    else:
        print(f"   ❌ Calibration is not a dict: {type(cal)}")
        cfg['calibration'] = {}
        fixed = True
else:
    print(f"\n⚠️  No calibration section found - creating empty one")
    cfg['calibration'] = {}
    fixed = True

# Fix classes_trigger if corrupted
if 'thresholds' in cfg:
    if 'classes_trigger' in cfg['thresholds']:
        classes = cfg['thresholds']['classes_trigger']
        print(f"\n🔍 Checking classes_trigger...")
        print(f"   Type: {type(classes)}")
       
        if isinstance(classes, dict):
            print("   ❌ classes_trigger corrupted (dict instead of list)")
            print(f"   Corrupted data: {classes}")
            cfg['thresholds']['classes_trigger'] = ['person']
            print("   ✅ Fixed: Set to ['person']")
            fixed = True
        elif isinstance(classes, list):
            print(f"   ✅ classes_trigger format is correct: {classes}")
        else:
            print(f"   ❌ Unexpected type: {type(classes)}")
            cfg['thresholds']['classes_trigger'] = ['person']
            print("   ✅ Fixed: Set to ['person']")
            fixed = True

# Check other threshold values
if 'thresholds' in cfg:
    print(f"\n📊 Threshold values:")
    for key, value in cfg['thresholds'].items():
        print(f"   {key}: {value} (type: {type(value).__name__})")

# Save if fixed
if fixed:
    print("\n💾 Saving fixed configuration...")
   
    # Create backup first
    backup_path = CONFIG_PATH.parent / 'config.yaml.broken'
    with open(CONFIG_PATH, 'r') as f:
        backup_content = f.read()
    with open(backup_path, 'w') as f:
        f.write(backup_content)
    print(f"   📦 Backup saved to: {backup_path}")
   
    # Save fixed config
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
   
    print(f"   ✅ Fixed config saved to: {CONFIG_PATH}")
    print("\n⚠️  NOTE: If ROIs were corrupted, please use ROI Editor to redraw them")
else:
    print("\n✅ Config is already in correct format, no fixes needed")

print("\n" + "=" * 50)
print("✅ Done!")
print("\nNext steps:")
print("1. Restart Flask app")
print("2. If ROIs were lost, use ROI Editor to redraw them")
print("3. Config Editor should now work correctly")