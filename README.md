# RustyRatracerr
🧪Experiments into HPC ray tracing using Rust.

## About
After having built a 4 node 32 core cluster at home using ex enterprise components I wanted to experiment with high performance computing and needed a computationally expensive task to throw at it.

I decided to follow [Ray Tracing In One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) whilst learning Rust, both are new to me so if you have any feedback I'd highly appreciate it!

[Follow this guide to get slurm & openMpi setup on your cluster](https://glmdev.medium.com/building-a-raspberry-pi-cluster-784f0df9afbd)

```bash
cargo build --release
sbatch ./run.sh
squeue
```

<p align="center">
   <div style="width:640;height:320">
       <img style="width: inherit" src="https://github.com/Timmoth/RustyRatracerr/blob/main/images/raytrace-1920x1080.png?raw=true">
</div>
</p>
