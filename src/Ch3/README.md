## Folder structure
```
src/Ch3/
├── build_dialect.sh    # The script to build Toy dialect with CMake
├── toy/
│   ├── c/        # Headers and impls of Toy dialect C-API
│   └── cpp/      # Headers and impls of Toy dialect
├── build_toy/    # (generated after building)
│   └── toy/cpp/  # - Files generated by tablegen
└── inst_toy/     # (generated after building)
    └── lib/      # - Libraries of Toy dialect and C-API
```
