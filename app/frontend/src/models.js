// Shared by AnalyzePage and BatchPage. These lists were duplicated, which is how
// the two screens ended up with different defaults and different badges.
//
// Descriptions state measured results only. The models fail on generators absent
// from their training data, but the mechanism is not established, since no GAN
// family was held out, so nothing here claims a cause.
export const MODELS = [
  { key: 'cnn', label: 'CNN', sub: 'Spatial features, ResNet-50' },
  { key: 'fft', label: 'FFT', sub: 'Frequency bands, generalises best to unseen generators' },
  { key: 'hybrid', label: 'Hybrid', sub: 'CNN and FFT features fused' },
  { key: 'stm', label: 'STM', sub: 'Handcrafted features, runs without a GPU' },
]

export const DEFAULT_MODEL = 'cnn'
