"""Reconstruct sample-ID-aligned RNA-seq matrices from the original CSV + RData.

Source CSV counts have no ID column. Their matching RData data.frame row.names
contain the exact sample IDs. Never align source expression and metadata by row position.

All accepted canonical files are verified cell-for-cell, without normalisation,
filtering, imputation or any change to gene counts. Existing canonical ZIP is
read-only; new rebuilt data are written under a run directory.
"""
from __future__ import annotations
import ctypes
import ctypes.util
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from .rdata_xdr_reader import read_ibd_rdata, pairlist_dict

EXPECTED_CANONICAL = "8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74"
EXPECTED_SAMPLES = {"d1":260,"d2":179,"d3":76,"d4":174}
EXPECTED_PREALIGNED = {"d1":1,"d2":2,"d3":0,"d4":16}


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()


def extract_ibd_rar(source: Path, directory: Path) -> None:
    """Extract only the eight expected files with system libarchive (RAR5 support).

    On platforms without libarchive, extract the RAR separately using 7-Zip and
    call reconstruct(..., extracted_csv_dir=...). No shell commands are run.
    """
    loc=ctypes.util.find_library("archive")
    if not loc:
        raise RuntimeError("libarchive missing: install libarchive or extract RAR5 with 7-Zip first")
    api=ctypes.CDLL(loc)
    def function(name,typ,args):
        fn=getattr(api,name);fn.restype=typ;fn.argtypes=args;return fn
    ptr=ctypes.c_void_p
    new=function("archive_read_new",ptr,[])
    filters=function("archive_read_support_filter_all",ctypes.c_int,[ptr])
    formats=function("archive_read_support_format_all",ctypes.c_int,[ptr])
    op=function("archive_read_open_filename",ctypes.c_int,[ptr,ctypes.c_char_p,ctypes.c_size_t])
    header=function("archive_read_next_header",ctypes.c_int,[ptr,ctypes.POINTER(ptr)])
    pathname=function("archive_entry_pathname",ctypes.c_char_p,[ptr])
    read=function("archive_read_data",ctypes.c_ssize_t,[ptr,ptr,ctypes.c_size_t])
    free=function("archive_read_free",ctypes.c_int,[ptr])
    a=new()
    expected={f"{p}_d{i}.csv" for p in ("count","desc") for i in range(1,5)}
    found=set()
    try:
        filters(a);formats(a)
        if op(a,str(source).encode(),10240)!=0:
            raise RuntimeError("RAR5 could not be opened by libarchive")
        entry=ptr();buf=ctypes.create_string_buffer(2**20)
        directory.mkdir(parents=True,exist_ok=True)
        while True:
            code=header(a,ctypes.byref(entry))
            if code==1:break
            if code!=0:raise RuntimeError("RAR5 header read failed")
            name=pathname(entry).decode("utf8")
            if name not in expected or name in found:
                raise ValueError(f"Unexpected or repeated original RAR entry: {name}")
            found.add(name)
            with (directory/name).open("wb") as output:
                while True:
                    n=read(a,buf,len(buf))
                    if n<0:raise RuntimeError(f"RAR5 read failed: {name}")
                    if not n:break
                    output.write(buf.raw[:n])
        if found!=expected:
            raise ValueError(f"Missing source CSV entries: {sorted(expected-found)}")
    finally:
        free(a)


def reconstruct(original_csv_dir: Path, original_rdata: Path, accepted_zip: Path,
                output_dir: Path, save_rebuilt_zip: bool=True) -> dict:
    """Run independent full-matrix equality checks and rebuild the aligned ZIP.

    Output counts/metadata retain original gene columns, including 41,257 d4 genes.
    The 9,983-gene intersection is a subsequent modeling-stage decision.
    """
    original_csv_dir=Path(original_csv_dir)
    output_dir=Path(output_dir)
    accepted_zip=Path(accepted_zip)
    if not accepted_zip.is_file() or digest(accepted_zip)!=EXPECTED_CANONICAL:
        raise ValueError("Accepted aligned ZIP missing or SHA-256 mismatch; do not replace")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output_dir}")
    output_dir.mkdir(parents=True,exist_ok=True)
    r=read_ibd_rdata(original_rdata)
    rows=[];audit=[]
    all_ids=set()
    rebuilt=output_dir/'reconstructed_counts_and_metadata.zip'
    with ZipFile(accepted_zip) as reference, ZipFile(rebuilt,"w",ZIP_DEFLATED,compresslevel=6) as writer:
        for d,expected_n in EXPECTED_SAMPLES.items():
            counts_obj=r[f"count_{d}"]
            names=pairlist_dict(counts_obj["attrs"])["names"]
            ids=pairlist_dict(counts_obj["attrs"])["row.names"]
            original=pd.read_csv(original_csv_dir/f"count_{d}.csv",dtype=np.int32)
            if original.shape!=(expected_n,len(names)) or original.columns.tolist()!=names:
                raise ValueError(f"{d}: original count columns differ from RData")
            arr=original.to_numpy(dtype=np.int32)
            rmat=np.column_stack(counts_obj["data"]).astype(np.int32)
            if not np.array_equal(arr,rmat):
                raise ValueError(f"{d}: original CSV count values differ from RData")
            desc=pd.read_csv(original_csv_dir/f"desc_{d}.csv",dtype={"ID":str})
            ref_desc=pd.read_csv(reference.open(f"desc_{d}.csv"),dtype={"ID":str})
            pd.testing.assert_frame_equal(desc,ref_desc,check_dtype=False)
            rdesc=r[f"desc_{d}"]
            rdesc_attr=pairlist_dict(rdesc["attrs"])
            rids=rdesc["data"][rdesc_attr["names"].index("ID")]
            if rids!=desc.ID.tolist() or rdesc_attr["row.names"]!=rids:
                raise ValueError(f"{d}: RData metadata row names and CSV IDs differ")
            class_r=rdesc["data"][rdesc_attr["names"].index("class")]
            if isinstance(class_r,dict):
                levels=pairlist_dict(class_r["attrs"])["levels"]
                labels=[levels[int(index)-1] for index in class_r["data"]]
            else:
                labels=class_r
            if labels!=desc["class"].tolist():
                raise ValueError(f"{d}: clinical labels differ from original RData")
            if len(ids)!=expected_n or len(set(ids))!=expected_n or len(set(desc.ID))!=expected_n or set(ids)!=set(desc.ID):
                raise ValueError(f"{d}: missing or duplicate sample IDs")
            if all_ids.intersection(ids):raise ValueError("Sample ID shared by different cohorts")
            all_ids.update(ids)
            old_pos={sample_id:i for i,sample_id in enumerate(ids)}
            indices=np.array([old_pos[sample_id] for sample_id in desc.ID],dtype=int)
            aligned=original.iloc[indices].reset_index(drop=True)
            aligned.insert(0,"ID",desc.ID.to_numpy())
            canonical=pd.read_csv(reference.open(f"count_{d}_aligned.csv"),dtype={"ID":str})
            if canonical.columns.tolist()!=aligned.columns.tolist() or canonical.ID.tolist()!=desc.ID.tolist():
                raise ValueError(f"{d}: canonical aligned gene names or ID order mismatch")
            if not np.array_equal(canonical.iloc[:,1:].to_numpy(dtype=np.int32),aligned.iloc[:,1:].to_numpy(dtype=np.int32)):
                raise ValueError(f"{d}: any reconstructed count value differs from canonical")
            already=int(np.count_nonzero(indices==np.arange(expected_n)))
            if already!=EXPECTED_PREALIGNED[d]:
                raise ValueError(f"{d}: unexpected positional-match count")
            for m,(sample_id,rowindex) in enumerate(zip(desc.ID,indices),start=1):
                audit.append(dict(dataset=d,sample_id=sample_id,original_count_row=int(rowindex+1),metadata_row=m,
                                  study=str(desc.Study.iloc[m-1]),diagnosis=str(desc['class'].iloc[m-1]),
                                  position_already_matched=bool(rowindex==m-1)))
            writer.writestr(f"count_{d}_aligned.csv",aligned.to_csv(index=False))
            writer.writestr(f"desc_{d}.csv",desc.to_csv(index=False))
            rows.append(dict(cohort=d,samples=expected_n,genes=len(names),already_aligned=already,
                             reordered=expected_n-already,labels_CD=int((desc['class']=='CD').sum()),
                             labels_Not_IBD=int((desc['class']=='Not IBD').sum()),
                             original_counts_match_RData=True,accepted_values_match=True,accepted_metadata_match=True))
            print(f"ALIGNMENT_VERIFIED={d} SAMPLES={expected_n} GENES={len(names)} FIXED={expected_n-already}",flush=True)
        audit_df=pd.DataFrame(audit)
        reference_audit=pd.read_csv(reference.open('sample_alignment_audit.csv'))
        pd.testing.assert_frame_equal(audit_df,reference_audit,check_dtype=False)
        writer.writestr('sample_alignment_audit.csv',audit_df.to_csv(index=False))
    pd.DataFrame(rows).to_csv(output_dir/'cohort_alignment_summary.csv',index=False)
    audit_df.to_csv(output_dir/'source_row_alignment_audit.csv',index=False)
    report=dict(status="PASS_ORIGINAL_RDATA_CSV_AND_ACCEPTED_SEMANTIC_RECONSTRUCTION",
                canonical_sha256=EXPECTED_CANONICAL,
                original_rdata_sha256=digest(original_rdata),
                rebuilt_zip_sha256=digest(rebuilt),
                rebuilt_is_semantically_identical=True,
                rebuilt_zip_byte_identity_required=False,
                total_samples=sum(EXPECTED_SAMPLES.values()),
                already_aligned=sum(EXPECTED_PREALIGNED.values()),
                requiring_reorder=sum(EXPECTED_SAMPLES.values())-sum(EXPECTED_PREALIGNED.values()),
                cohorts=rows,finished_utc=datetime.now(timezone.utc).isoformat())
    (output_dir/'report.json').write_text(json.dumps(report,indent=2))
    if not save_rebuilt_zip:rebuilt.unlink()
    return report


def run_from_root(root: Path,output_dir: Path,extracted_csv_dir: Path|None=None, save_rebuilt_zip:bool=False) -> dict:
    root=Path(root)
    source=root/'data/source'
    csv_dir=Path(extracted_csv_dir) if extracted_csv_dir else None
    if csv_dir is None:
        with TemporaryDirectory(prefix="cmps355_source_csv_") as tmp:
            extract_ibd_rar(source/'ibd_datasets.rar',Path(tmp))
            report=reconstruct(Path(tmp),source/'ibd_datasets.RData',root/'data/validated/ibd_dataset_sample_aligned.zip',output_dir,save_rebuilt_zip)
    else:
        report=reconstruct(csv_dir,source/'ibd_datasets.RData',root/'data/validated/ibd_dataset_sample_aligned.zip',output_dir,save_rebuilt_zip)
    report['original_rar_sha256']=digest(source/'ibd_datasets.rar')
    (Path(output_dir)/'report.json').write_text(json.dumps(report,indent=2))
    return report
