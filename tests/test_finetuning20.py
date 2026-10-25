from pathlib import Path
import torch
from amc.config import load_config
from amc.finetuning20 import migrate_classifier
from amc.model import build_model


def test_migration_matches_names_after_reordering_and_preserves_new_rows():
    root=Path(__file__).resolve().parents[1]
    old=load_config(root/"configs/baseline.yaml")
    new=load_config(root/"configs/amc20_v1.yaml")
    new["classes"]=list(reversed(new["classes"]))
    torch.set_num_threads(2)
    torch.manual_seed(17)
    source=build_model(old).eval()
    payload={"classes":old["classes"],"config":old,"model_state":source.state_dict()}
    torch.manual_seed(23)
    initial=build_model(new)
    torch.manual_seed(23)
    target,report=migrate_classifier(payload,new)
    target.eval()
    x=torch.randn(2,2,2048)
    with torch.inference_mode():
        a=source(x)
        b=target(x)[:,[new["classes"].index(n) for n in old["classes"]]]
    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    for name in report["new_classes"]:
        i=new["classes"].index(name)
        assert torch.equal(target.classifier.weight[i],initial.classifier.weight[i])
    assert len(report["migrated_classes"])==13 and len(report["new_classes"])==7
