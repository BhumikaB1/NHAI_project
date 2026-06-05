if(NOT TARGET react-native-nitro-image::NitroImage)
add_library(react-native-nitro-image::NitroImage SHARED IMPORTED)
set_target_properties(react-native-nitro-image::NitroImage PROPERTIES
    IMPORTED_LOCATION "C:/Users/Bhumika/NHAI_project/node_modules/react-native-nitro-image/android/build/intermediates/cxx/Debug/c6upo1m6/obj/x86_64/libNitroImage.so"
    INTERFACE_INCLUDE_DIRECTORIES "C:/Users/Bhumika/NHAI_project/node_modules/react-native-nitro-image/android/build/headers/nitroimage"
    INTERFACE_LINK_LIBRARIES ""
)
endif()

