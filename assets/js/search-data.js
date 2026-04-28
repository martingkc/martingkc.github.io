// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "Things I&#39;ve built.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "post-stencil-in-cuda",
        
          title: "Stencil in CUDA",
        
        description: "Exploring different optimization methods in the implementation of 3D stencils in CUDA.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/scan/";
          
        },
      },{id: "post-convolutions-in-cuda",
        
          title: "Convolutions in CUDA",
        
        description: "Exploring different optimization methods in the implementation of Convolutions in CUDA.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/convolutions-in-cuda/";
          
        },
      },{id: "post-histogram-in-cuda",
        
          title: "Histogram in CUDA",
        
        description: "Exploring different optimization methods in the implementation of Histograms in CUDA.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/histogram_in_CUDA/";
          
        },
      },{id: "post-fakellava-building-a-lightweight-llava-style-vlm-with-qwen2-5-and-clip",
        
          title: "FakeLLaVA — Building a Lightweight LLaVa style VLM with Qwen2.5 and CLIP",
        
        description: "Training a lightweight LLaVA-style VLM by connecting a CLIP vision encoder to Qwen2.5-0.5B through a two-layer projection MLP.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/LLaVa/";
          
        },
      },{id: "post-clip-models-and-image-captioning-using-clip-embeddings",
        
          title: "CLIP Models and Image Captioning using CLIP embeddings.",
        
        description: "Exploring contrastive learning and Info-NCE loss and training a CLIP vision encoder, then building a GPT-2 image captioning model (ClipCap) trained on chest X-ray CLIP embeddings.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/image-captioning-using-clip/";
          
        },
      },{id: "post-learning-vit-39-s",
        
          title: "Learning VIT&#39;s",
        
        description: "Building a Vision Transformer (ViT) from scratch in PyTorch with 2D-RoPE positional encoding, applied to multi-label chest X-ray classification.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/learning-VIT/";
          
        },
      },{id: "post-hello-world",
        
          title: "Hello World",
        
        description: "First post.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/hello-world/";
          
        },
      },{id: "projects-brainfuck-compiler",
          title: 'Brainfuck Compiler',
          description: "Brainfuck-to-x86 compiler written in C as a low-level systems exercise, covering lexing, IR generation, and native code emission.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/10_project/";
            },},{id: "projects-idata",
          title: 'iData',
          description: "Self-hosted document intelligence platform with a multi-agent RAG backend. Ingests files from disk or cloud storage, parses and embeds them, then answers natural-language questions with citations, SQL lookups, and charts.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-ambercow-com",
          title: 'ambercow.com',
          description: "News aggregation platform that scrapes and clusters content from 50+ international outlets and generates summaries and Q&amp;A through a custom RAG pipeline.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-fakellava",
          title: 'FakeLLaVA',
          description: "Lightweight LLaVA-style VLM built by connecting a CLIP ViT-B/32 encoder to Qwen2.5-0.5B through a two-layer projection MLP. Trained in two stages on CC3M and LLaVA-Instruct-150K.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-cot-rag-retriever-agent",
          title: 'COT-RAG Retriever Agent',
          description: "Language model fine-tuned with Unsloth to act as a chain-of-thought retriever inside a RAG pipeline, enabling structured reasoning over retrieved passages before answering.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-d-mel-tts-fine-tuning",
          title: 'D-Mel TTS Fine-Tuning',
          description: "Data generation pipeline that downloads YouTube videos, performs speaker diarization, and converts audio into D-Mel tokens for TTS fine-tuning. Evaluated LoRA-based and full fine-tuning approaches.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-voice-extraction-amp-dataset-builder",
          title: 'Voice Extraction &amp;amp; Dataset Builder',
          description: "Pipeline that diarizes audio, transcribes each speaker segment with Whisper, and packages the output as a labeled dataset ready for TTS or ASR fine-tuning.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-ann-oy-vector-search-engine",
          title: 'ANN-OY Vector Search Engine',
          description: "On-disk vector search engine in C with cosine-similarity search over large embedding datasets. Uses SQLite3 for ACID-compliant metadata and binary serialization for persistent index reuse.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-risc-v-simulator",
          title: 'RISC-V Simulator',
          description: "RISC-V instruction set simulator in Python covering arithmetic, logical, shift, memory, and branch instructions, with an interactive Streamlit UI for step-through execution.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6D%61%72%74%69%6E%67%6F%6B%63%75@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/martingkc", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/martin-gokcu", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
