# Cerebras Wafer-Scale Cluster

Exa-scale performance, single device simplicity

## Accelerate AI Model Training

The Cerebras Wafer-Scale Cluster (WSC) is a revolutionary technology suite that efficiently handles the enormous computational needs of AI model training. It centers around the CS-3 system, powered by the 3rd generation Wafer-Scale Engine (WSE-3)—the world’s largest AI-optimized processor. The WSC integrates MemoryX for high-capacity, off-chip model weight storage, and SwarmX for effective weight broadcasting and gradient reduction across the cluster. This setup allows the WSC to adeptly train multi-trillion parameter models, achieving near perfect linear-scaled performance and simplifying the complexity seen in traditional distributed computing.

## Powered by the 3rd Generation Wafer-Scale Engine

The Cerebras WSE-3 is 46,250 square millimeters of silicon, 4 trillion transistors, 900K cores, 44 GB on-chip memory, and delivers an unparalleled 125 petaFLOPS of AI compute. It surpasses all other processors in AI-optimized cores, memory speed, and on-chip fabric bandwidth.

## Simplifying Large-Scale AI Computing

Conventional systems struggle to scale, hampered by the challenges of synchronizing vast arrays of processors across many nodes. The Cerebras WSC thrives, seamlessly integrating its components for large-scale, parallel computation and providing a straightforward, data-parallel programming interface.

## Specs

* 900,000 compute cores
* 125 PetaFLOPs of AI Performance
* 44 GB on-chip memory
* 12 to 1,200 TB of off-chip model memory
* 21 PB/sec memory bandwidth
* 214 PB/sec core-to-core bandwidth

## AI Supercomputers

Condor Galaxy (CG), the supercomputer built by G42 and Cerebras, is the simplest and fastest way to build AI models in the cloud. With over 16 ExaFLOPs of AI compute, Condor Galaxy trains the most demanding models in hours rather than days. The terabyte scale MemoryX system natively accommodates 100 billion+ parameter models, making large scale training simple and efficient.

| Cluster  | ExaFLOPs | Systems  | Memory |
| -------- | -------- | -------- | ------ |
| CG1      | 4        | 64 CS-2s | 82 TB  |
| CG2      | 4        | 64 CS-2s | 82 TB  |
| CG3      | 8        | 64 CS-3s | 108 TB |
