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
      },{
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
