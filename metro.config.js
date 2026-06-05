
const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');

const config = {
  projectRoot: __dirname,
  watchFolders: [
    __dirname,
  ],
  resolver: {
    blacklistRE: /node_modules\/react-native-nitro-modules\/android\/.cxx\/.*/,
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
