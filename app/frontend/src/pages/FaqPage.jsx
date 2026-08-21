// Figures here are quoted from research/results/ and should be changed only
// alongside that data. The held-out test set is 2,345 images, the sum of the
// three cumulative stage test splits, 82 + 757 + 1,506.
//
// Nothing in this file explains why the models fail on unseen generators. The
// training data is two thirds diffusion, so the older GAN-versus-diffusion
// account is wrong, and no GAN family was held out, so the alternative is not
// established either. State the measurement, not the mechanism.

const DISCLAIMER = [
  "No detector is reliable enough to treat as proof. Treat a verdict as one piece of evidence to weigh, not as an answer.",
  "On images resembling its training data the strongest model is correct about 84% of the time. On images from generators it has never seen, most of the models are close to chance or worse.",
  "Heavily compressed, resized, screenshotted or already-degraded images give less reliable results, because re-encoding damages the traces the models rely on.",
]

const SECTIONS = [
  {
    title: 'How to read a result',
    items: [
      {
        q: 'What is the P(AI) score?',
        a: 'It is the probability the model assigns to the image being AI-generated. Zero means the model is confident the image is authentic, one means it is confident the image is AI. The verdict flips at 0.5.',
      },
      {
        q: 'What does confidence mean?',
        a: 'Confidence is the probability of whichever side won, so it never falls below 50%. A verdict at P(AI) = 0.5 carries 50% confidence, and P(AI) = 0.95 carries 95%. Confidence and P(AI) are different numbers and only agree when the verdict is AI.',
      },
      {
        q: 'What does the tool actually do?',
        a: 'It runs one detection model over the image you upload and reports a verdict, a probability and supporting visualisations. It does not search the web, check metadata, or look for a known source.',
      },
    ],
  },
  {
    title: 'Choosing a model',
    items: [
      {
        q: 'Which model should I use?',
        a: 'There is no safe default. CNN and Hybrid score highest on images resembling the training data, and detect almost nothing from generators absent from it. FFT is the weakest in that first setting and by far the strongest in the second. STM sits between the two and needs no GPU. Run more than one model and compare: when they disagree, that disagreement is itself informative, and it is the most honest signal this tool can give you.',
      },
      {
        q: 'What if two models disagree?',
        a: 'Take it as a warning that the image is outside what these models handle confidently. It happens most often on images from recent generators, and on content that is neither a photograph nor AI output, such as renders and screenshots.',
      },
      {
        q: 'Why does STM take longer?',
        a: 'STM extracts 1,822 features on the CPU (edge gradients, micro-textures, frequency coefficients, colour statistics and noise residuals) before passing them to LightGBM. That is slower than a single GPU forward pass, and it is also why STM runs on machines without a GPU.',
      },
    ],
  },
  {
    title: 'Why unseen generators are hard',
    items: [
      {
        q: 'What was the training data?',
        a: 'Six AI sources in equal share: ForenSynths, CIFAKE, and four GenImage sets covering BigGAN, MidJourney, Stable Diffusion 1.4 and Stable Diffusion 1.5. The authentic half is photographs, drawn from COCO, ImageNet, Unsplash and Flickr30k. Every generator in that list predates the current wave of image models.',
      },
      {
        q: 'What happens on newer generators?',
        a: 'Performance collapses for the spatial models. Tested against 10,000 images from 46 generators none of the models had seen, including Flux, Imagen, DALL-E 3, SDXL and recent MidJourney versions, CNN and Hybrid fell to near zero while FFT held up.',
      },
      {
        q: 'Why does that happen?',
        a: 'Not established. The obvious explanation, that the training set is GAN-heavy and modern output is diffusion, does not survive contact with the data: two thirds of the AI training images are already diffusion. Isolating the cause would need a held-out GAN family to compare against, which this project did not run. The collapse is a measured result here, not an explained one.',
      },
    ],
  },
  {
    title: 'Results on unseen generators',
    items: [
      {
        q: 'What are the actual numbers?',
        a: 'Against 10,000 images from generators absent from training, all of them AI-generated, the proportion each model correctly flagged was: FFT 85.2%, STM 45.4%, CNN 2.25%, Hybrid 1.38%. The ordering is close to the reverse of the ordering on the held-out test set.',
      },
      {
        q: 'Is that a bug?',
        a: 'No. It is a measured finding of this project and it is the reason the interface exposes all four models instead of picking the one with the best headline score. Two later variants of Hybrid were trained to test possible fixes; neither recovered the gap.',
      },
    ],
  },
  {
    title: 'Visualisations',
    items: [
      {
        q: 'What do the visualisations show?',
        a: 'CNN and Hybrid show a Grad-CAM heatmap of the regions that most influenced the verdict. FFT and Hybrid show a frequency spectrogram of how energy is distributed across the image spectrum. STM shows how much each of its five feature groups contributed to this particular decision.',
      },
      {
        q: 'Do the image property panels decide the verdict?',
        a: 'No. Contrast, sensor noise and the RGB distribution are computed directly from the image and are shown for context. No model reads them, with one exception: STM does use noise residuals and colour statistics as inputs, so for STM they are genuine evidence.',
      },
    ],
  },
]

const MODELS = [
  {
    name: 'CNN',
    full: 'Spatial CNN',
    speed: 'Fast',
    auc: '93.8%',
    desc: 'ResNet-50 backbone trained to detect spatial irregularities in the pixel domain. Second on accuracy on the held-out test set at 84.4%, and detects 2.25% of images from unseen generators.',
  },
  {
    name: 'FFT',
    full: 'Frequency FFT',
    speed: 'Fast',
    auc: '67.1%',
    desc: 'Analyses four learned radial frequency bands in the Fourier spectrum. Weakest of the four on the held-out test set, and the only one that holds up on generators absent from training, where it detects 85.2%.',
  },
  {
    name: 'Hybrid',
    full: 'Hybrid Fusion',
    speed: 'Slower',
    auc: '93.9%',
    desc: 'Combines 2048-dim CNN spatial features and 256-dim FFT spectral features through a learned fusion network. Leads on AUC and F1 on the held-out test set, and detects the least of any model on unseen generators at 1.38%.',
  },
  {
    name: 'STM',
    full: 'Handcrafted STM',
    speed: 'Slowest',
    auc: '83.6%',
    desc: '1,822 handcrafted features (HOG, LBP, DCT, colour statistics, noise residual) classified by LightGBM. No neural network, runs without a GPU, and detects 45.4% on unseen generators.',
  },
]

const METRICS = [
  { name: 'P(AI)', desc: 'Probability the image is AI-generated. Below 0.5 gives an authentic verdict, above 0.5 gives an AI verdict.' },
  { name: 'Confidence', desc: 'The probability of whichever side won, so it runs from 50% at the boundary to 100% at either extreme. Not the same as P(AI).' },
  { name: 'AUC-ROC', desc: 'Threshold-independent ranking quality. 1.0 is perfect, 0.5 is chance.' },
  { name: 'Contrast', desc: 'Standard deviation of pixel brightness, scaled by 100. Photographs commonly fall between 30 and 70.' },
  { name: 'Sensor Noise', desc: 'Standard deviation of the high-frequency residual after subtracting a blurred copy, scaled by 100. Photographs commonly fall between 8 and 25.' },
]

const PRIVACY = [
  {
    q: 'Are my images saved?',
    a: 'No. Images are held in memory for the duration of the request and discarded afterwards. They are never written to disk or to the database.',
  },
  {
    q: 'What is stored?',
    a: 'Five things per analysis: the filename, the model used, the verdict, the P(AI) score and a timestamp. The score stored is P(AI), not confidence.',
  },
  {
    q: 'Can I delete my history?',
    a: 'Not from the interface. Filenames are retained in the database indefinitely and there is no deletion mechanism, so avoid uploading files whose names you would not want kept.',
  },
]

const DATA = [
  {
    q: 'How were the models tested?',
    a: 'Trained across three cumulative stages totalling 15,500 images, and evaluated on a held-out test set of 2,345 images that the models never saw during training. Every figure quoted in this application comes from that evaluation.',
  },
  {
    q: 'What kinds of images can I upload?',
    a: 'JPEG, PNG or WebP, up to 10MB. Be aware that the authentic half of the training data is photographs only. Screenshots, scans, renders, vector art and other synthetic-but-not-AI images fall outside both training classes, so verdicts on them are unreliable in a way the confidence score does not capture.',
  },
]

function QA({ q, a }) {
  return (
    <div className="py-3 border-b border-cappuccino/30 last:border-0">
      <p className="text-sm font-bold text-espresso mb-1">{q}</p>
      <p className="text-xs text-roast leading-relaxed">{a}</p>
    </div>
  )
}

function Section({ n, title, children }) {
  return (
    <div className="card p-6">
      <p className="text-lg font-black text-espresso mb-4">
        <span className="text-cappuccino mr-2">{n}</span>{title}
      </p>
      {children}
    </div>
  )
}

export default function FaqPage() {
  return (
    <div className="max-w-3xl mx-auto space-y-4">
      <h1 className="page-title">Help and Reference</h1>

      <div className="card p-6 border-caramel/40 bg-latte">
        <p className="text-lg font-black text-espresso mb-3">
          <span className="text-cappuccino mr-2">1</span>Before you start
        </p>
        <div className="space-y-2">
          {DISCLAIMER.map((line, i) => (
            <p key={i} className="text-xs text-roast leading-relaxed">{line}</p>
          ))}
        </div>
      </div>

      {SECTIONS.map((s, i) => (
        <Section key={s.title} n={i + 2} title={s.title}>
          <div className="space-y-0">
            {s.items.map((item) => <QA key={item.q} q={item.q} a={item.a} />)}
          </div>
        </Section>
      ))}

      <Section n={7} title="Model reference">
        <p className="text-xs text-roast leading-relaxed mb-4">
          Trained on 15,500 images across six AI sources and evaluated on a held-out test set of 2,345 images. AUC is from Stage 3. Detection rates are against 10,000 images from generators absent from training.
        </p>
        <div className="space-y-0">
          {MODELS.map((m) => (
            <div key={m.name} className="py-3 border-b border-cappuccino/30 last:border-0">
              <div className="flex items-center justify-between mb-1">
                <p className="text-sm font-black text-espresso">
                  {m.name}
                  <span className="text-xs font-normal text-roast ml-2">{m.full}</span>
                </p>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-roast">AUC {m.auc}</span>
                  <span className="text-xs font-semibold text-roast bg-cream border border-cappuccino/60 rounded-md px-2 py-0.5">{m.speed}</span>
                </div>
              </div>
              <p className="text-xs text-roast leading-relaxed">{m.desc}</p>
            </div>
          ))}
        </div>
        <p className="text-sm font-bold text-espresso mt-5 mb-2">Metrics</p>
        <div className="space-y-0">
          {METRICS.map((m) => (
            <div key={m.name} className="py-2.5 border-b border-cappuccino/30 last:border-0">
              <p className="text-sm font-bold text-espresso">{m.name}</p>
              <p className="text-xs text-roast mt-0.5 leading-relaxed">{m.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section n={8} title="Data and privacy">
        <div className="space-y-0">
          {DATA.map((item) => <QA key={item.q} q={item.q} a={item.a} />)}
          {PRIVACY.map((item) => <QA key={item.q} q={item.q} a={item.a} />)}
        </div>
      </Section>
    </div>
  )
}
