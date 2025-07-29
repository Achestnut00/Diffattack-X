

<h1>Diffattack-X: An Effective Transferable Adversarial Attack
Based on Diffusion Models</h1>



### After the paper is accepted, we will further improve the code in detail


## Abstract

![Fig1 overview](https://github.com/user-attachments/assets/81a5e3e0-433c-4a01-9d55-ce779ce5ebdd)

Deep learning models are highly susceptible to adversarial attacks. Existing methods often oper-ate in the RGB (Red, Green, and Blue) space, relying on perturbations of the Lp-norm that are
perceptible and limited in transferability to black-box models. Moreover, attack methods manipu-lating latent variables in diffusion models typically struggle to enhance attack performance against
black-box models significantly or to balance attack efficacy and visual imperceptibility effectively. We propose DiffAttack-X, an effective diffusion model-based adversarial attack method to address these challenges. To enhance attack effectiveness, bi-level routing attention is introduced to reduce the cor-
relation between adversarial samples and their corresponding correct labels. Furthermore, focal loss
enhances focus on small-object features, increasing classification difficulty. For visual imperceptibility,
group squeeze-and-excitation attention and self-attention mechanisms maintain structural integrity
across spatial and channel dimensions, and adaptive mean squared error loss constrains semantic
deviations at the pixel level, preserving visual similarity. Experiments on the ImageNet dataset show
that DiffAttack-X outperforms existing methods, with performance gains of 5.4% and 10.3% across
11 black-box transfer scenarios and four adversarially trained models, along with notable improve-
ments under two purification methods. Visualizations further validate the superiority of DiffAttack-X
in attack effectiveness and stealth.


### In terms of project configuration and baseline code, we used Diffattack's operations, please refer to <a>https://github.com/WindVChen/DiffAttack</a>


## Requirements

1. Hardware Requirements
    - GPU: 1x high-end NVIDIA GPU with at least 16GB memory

2. Software Requirements
    - Python: 3.8
    - CUDA: 11.3
    - cuDNN: 8.4.1

   To install other requirements:

   ```
   pip install -r requirements.txt
   ```

3. Datasets
   - There have been demo-datasets in [demo](demo), you can directly run the optimization code below to see the results.
   - If you want to test the full `ImageNet-Compatible` dataset, please download the dataset [ImageNet-Compatible](https://drive.google.com/file/d/1sAD1aVLUsgao1X-mu6PwcBL8s68dm5U9/view?usp=sharing) and then change the settings of `--images_root` and `--label_path` in [main.py](main.py)

4. Pre-trained Models
   - We adopt `Stable Diffusion 2.0` as our diffusion model, you can load the pretrained weight by setting `--pretrained_diffusion_path="stabilityai/stable-diffusion-2-base"` in [main.py](main.py).
   - For the pretrained weights of the adversarially trained models (Adv-Inc-v3, Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub>) in Section 4.2.2 of our paper, you can download them from [here](https://github.com/ylhz/tf_to_pytorch_model) and then place them into the directory `pretrained_models`.

5. (Supplement) Attack **CUB_200_2011** and **Standford Cars** datasets
   - Dataset: Aligned with **ImageNet-Compatible**, we randomly select 1K images from **CUB_200_2011** and **Standford Cars** datasets, respectively. You can download the dataset here [[CUB_200_2011](https://drive.google.com/file/d/1umBxwhRz6PIG6cli40Fc0pAFl2DFu9WQ/view?usp=sharing) | [Standford Cars](https://drive.google.com/file/d/1FiH98QyyM9YQ70PPJD4-CqOBZAIMlWJL/view?usp=sharing)] and then change the settings of `--images_root` and `--label_path` in [main.py](main.py). Note that you should also set `--dataset_name` to `cub_200_2011` or `standford_car` when running the code.
   - Pre-trained Models: You can download models (ResNet50, SENet154, and SE-ResNet101) pretrained on CUB_200_2011 and Standford Cars from [Beyond-ImageNet-Attack](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack) repository. Then place them into the directory `pretrained_models`.

## Crafting Adversarial Examples

To craft adversarial examples, run this command:

```
python main.py --model_name <surrogate model> --save_dir <save path> --images_root <clean images' path> --label_path <clean images' label.txt>
```
The specific surrogate models we support can be found in `model_selection` function in [other_attacks.py](other_attacks.py). You can also leverage the parameter `--dataset_name` to generate adversarial examples on other datasets, such as `cub_200_2011` and `standford_car`.

The results will be saved in the directory `<save path>`, including adversarial examples, perturbations, original images, and logs.

For some specific images that distort too much, you can consider weaken the inversion strength by setting `--start_step` to a larger value, or leveraging pseudo masks by setting `--is_apply_mask=True`.

## Evaluation

### Robustness on other normally trained models

To evaluate the crafted adversarial examples on other black-box models, run:

```
python main.py --is_test True --save_dir <save path> --images_root <outputs' path> --label_path <clean images' label.txt>
```
The `--save_dir` here denotes the path to save only logs. The `--images_root` here should be set to the path of `--save_dir` in above [Crafting Adversarial Examples](#crafting-adversarial-examples).


## Results
<img width="2936" height="3918" alt="Fig5 visual" src="https://github.com/user-attachments/assets/49663b9d-f4d8-44be-8b26-f8ad5ca205a2" />
