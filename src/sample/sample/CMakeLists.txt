add_mlir_dialect_library(MLIRSample
    SampleDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/sample

    # DEPENDS
    # ...

    LINK_LIBS PUBLIC
    MLIRIR
)
