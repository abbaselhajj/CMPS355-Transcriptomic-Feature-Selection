# Notebook outputs and safe future GitHub release

A review of the **37** original Drive notebooks (2026-11-08) found that **31** had saved execution outputs. Many contain local Colab Drive paths, and two include historical traceback text. No `_Fresh` folder suffix was found in notebook **code cell sources**. Notebook execution outputs are useful private evidence and were **not bulk-deleted**.

The project's synthetic test suite originally failed two checks because `notebooks/00_setup.ipynb` and `notebooks/01_data_validation.ipynb` had execution outputs, contrary to the test contract. The two executed originals were moved to private `docs/history/` as `00_setup_executed_original_20261108.ipynb` and `01_data_validation_executed_original_20261108.ipynb`; only those two **public-facing** notebooks had their output and execution-count fields cleared. The source cells were preserved. On an independent local source snapshot, the full suite then **passed all 35 tests**.

**Do not directly copy every Drive file into GitHub.** `runs/` contains patient IDs, model objects and historical outputs, while other notebooks still embed private execution traces. For final publication, run the newly added safe-export script on a local copy of the Drive project:

```bash
python scripts/export_public_repository.py /path/to/CMPS355_Research /path/to/CMPS355_Public_Export
```

It creates a separate source/reports-only tree, strips notebook execution outputs **only in the exported copy**, and excludes the private dataset, `runs/`, native model archives, cached environments, proposal/course PDFs, and `docs/history/`. It refuses to overwrite an existing export and checks for patient IDs/token patterns in candidate text. You must still manually inspect the resulting Git diff, licenses, screenshots and notebook markdown before making it public. Test it with `python -m unittest discover -s tests -v` inside the export directory.
