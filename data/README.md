# Data availability and licensing

The code can download CSI 300 historical index data from the documented public
endpoint in `src/data_loader.py`.

Raw downloaded CSV files and processed modeling CSVs are excluded from the
public Git repository because provider redistribution terms may differ from the
software license. Their manifests remain in the repository to document source,
coverage, retrieval time, and checksums.

The MIT License applies to project code and original documentation. It does not
grant rights to third-party market data, provider trademarks, or external
research materials.

For reproducible local runs:

```powershell
.\run_research.cmd
```

If a provider changes access or terms, update the loader and data-source
documentation rather than committing an unauthorized data copy.

