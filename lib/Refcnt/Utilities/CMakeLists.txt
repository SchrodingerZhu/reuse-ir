add_library(
  MLIRRefcntUtilities
  INTERFACE
)

get_property(REFCNT_UTILITIES_ADT_HAS_ABSL GLOBAL PROPERTY REFCNT_UTILITIES_ADT_HAS_ABSL)
if (REFCNT_UTILITIES_ADT_HAS_ABSL)
  target_link_libraries(MLIRRefcntUtilities INTERFACE absl::btree)
endif()