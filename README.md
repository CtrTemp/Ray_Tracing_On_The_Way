# Ray_Tracing_On_The_Way
My Own Render Program

本工程参考Peter Shirley的《Ray Tracing in One Weekend》系列书目，在此基础上进行扩充。

## 现有实现：
1、基本几何体渲染：球体、立方体

2、支持基本常数表面texture、噪声texture

3、当前实现版本为最基础的whitted styled ray tracing，不加入任何加速结构、图像去噪

4、基本的表面材质硬编码定义：基本镜面、透明表面、漫反射表面

5、基本模型内加速结构

6、简单模型导入渲染（<=3000k）

7、实现全局bvh加速结构

8、直接向光源采样，正确实现基本的渲染方程

## 最近更新：

#### 1、程序以及注释进行了删减，之前冗余的部分被最大幅度删除，方便之后的调试优化（23/02/16）

  * 解决采样金属/玻璃材质半球均匀采样pdf错误的问题，这两种材质的采样角应远小于满反射表面，优化后能量传递更加真实准确（23/02/15）
  
  * 解决椒盐噪声问题，这是由于数值精度问题引起的scatter二次射线与原本生成二次射线的那个位点自相交的bug（23/02/14）

  * 基于现有的BRDF方程，扩充融入BTDF加以完善，从而可以适应透射表面材质：如玻璃等（23/02/13）

  * 直接向光源采样，正确应用rendering equation，抛弃Whitted styled Ray-Tracing（23/02/07）

  * 修复了加速渲染结构的bug，之前很多的黑色噪点现在已经得到了优化（23/02/04）
  * 加速渲染结构的渲染速度已经回归正常，不会再出现比一般暴力遍历算法更慢的情况（23/02/04）
  
  * 整个渲染函数，射线投射循环等现已全部被融入到camera类中，成为其成员函数，main函数被大幅简化。（23/01/11）
  * 对于组件的创建与拼接，参考Vulkan中的方式，通过配置特殊的创建结构体，对camera、framebuffer、renderpass等组件进行创建，后期会逐步统一（23/01/11）
  
  * 优化了类的层级结构，hitable下属四类：geometry基本几何、primitive基本面元、models面元几何（面元列表）、group组（对象列表）。（23/01/08）
    * 原sphere类以及box类被归入基本几何geometry类，并成为其派生类，后期将添加用于描述空间曲线/曲面的curve_function/surface_function类；（23/01/08）
    * 原triangle类被归为primitive的派生类，后期会添加四边形面元quadrangle类；（23/01/08）
    * triangleList被归为models，现在models类具有抽象primitive列表取代单一的三角形列表类；（23/01/08）
    * bvh加速结构也因此进行了重写。（23/01/08）

  * 全局加速结构实现与测试（23/01/06）

  * 引入了对单一模型的加速结构，使用层级包围盒结构构建树，但当前不支持对场景内多个单独物体的加速结构构建（23/01/05）

  * 支持对较为复杂模型的渲染，当前可渲染的模型面片数在1000～2000数量级（23/01/05）

  * 引入gdb辅助调试工具，现在支持调试（23/01/05）

  * 渲染了天空盒，并得到了基本验证（23/01/03）

  * 支持uv贴图，但当前不支持顶点复用也无法支持从模型导入贴图（23/01/03）

  * 支持基本模型导入并渲染（当前只支持.obj格式）（22/12/31）

  * 加入hitable新派生类 triangleList，作为model基础，支持从三角形顶点缓冲区+索引缓冲区两种基本构造方式（22/12/31）

  * 加入hitable新派生类 triangle，作为最基本的建模面元，可以渲染单个三角形（22/12/30）




## TODOs：
1、补充加速结构，模型三角形面元列表内部树状加速结构构建（BVH层级包围盒，按照面元数量进行空间等分）（**done**）

2、补充加速结构，整体世界坐标系中物体的树状加速结构构建（BVH层级包围盒，按照对象数量进行空间等分）（**done**）

3、构造并渲染天空球（**done**）

4、面内顶点插值采样器

5、UV贴图实现（**done**）

6、较复杂模型，从模型读取材质特征

7、树状加速结构优化以及精确测试

8、模型在场景中的变换（包括缩放以及平移等）

9、引入并使用标准矩阵运算库

10、正确应用渲染方程，向光源采样--适应lambertian表面材质（**done**）

11、正确应用渲染方程，向光源采样--适应mental表面材质（**done**）

12、正确应用渲染方程，向光源采样--适应dielectric表面材质（**done**）


## Gallery

#### 2023/02/16

Skybox test-Mental

![22-1_skybox_50spp_35s](https://user-images.githubusercontent.com/89559223/219355998-4204d3f0-2528-45a2-b72d-528c53be4fd9.png)


Skybox test-Glass

![23-2](https://user-images.githubusercontent.com/89559223/219356126-832f4532-ae30-4dae-a5cf-7c4638777aab.png)


Model test mental Bunny

![24-2_mental_15spp_70sec](https://user-images.githubusercontent.com/89559223/219356213-2b35dbc9-80f6-43d9-b78e-6e120b5f62c9.png)


Model test glass Bunny

![25-2_15spp_137sec](https://user-images.githubusercontent.com/89559223/219356281-f7e48b0a-a163-47cb-90bf-d57019fc6984.png)


Cornell Box-01-mirror

![26-2_5spp_29sec](https://user-images.githubusercontent.com/89559223/219356455-d3d04882-66d2-44c7-819c-c5390b2377c4.png)


Cornell Box-02-lambertian

![27-3_50spp_288sec](https://user-images.githubusercontent.com/89559223/219356388-c70aa210-aaca-4abf-80a6-d2d10e6cae67.png)



#### 2023/02/15

Maybe you've already mentioned that the reflect of a mental surface may have a too strong light power, even stronger than the surface itself. Like that: 

![04_50spp_why_lighter](https://user-images.githubusercontent.com/89559223/219352319-306fc911-9e80-4691-88ee-0d7207b3c8d9.png)


That is caused by a wrong sample pdf function. We know that we can use a pure uniform sample function to sample lambertian surface, which the pdf=1/S=1/(2*pi). Of course, the mental and dielectric surface can still use a uniform sample function, however, the S is not equals to 2*pi! That is to say, the sample angle must be much less than that for lambertian surface. Then, we almost get the correct result:

![15_mental_pdf_25percents_50spp](https://user-images.githubusercontent.com/89559223/219352592-d3c79605-bcf2-46ab-8fb4-031cd3bf8a65.png)

If we allow indirect light scattered to direct light source for mental and dielectric surface, then we can get a even better highlight view:

![17_highLight_optimize](https://user-images.githubusercontent.com/89559223/219355193-3d6466e1-864a-4ce4-b0bb-99c8f70a9edd.png)

![21-2](https://user-images.githubusercontent.com/89559223/219355435-47de2fdc-b8ee-4e0f-b317-0d20cd45795d.png)

Mirror Ground:

![20_pure_mirror](https://user-images.githubusercontent.com/89559223/219355321-ab17a3f1-2039-4617-ba6c-23b10b7f6a08.png)




#### 2024/02/14

You should have mentioned Salt Pepper noise spread all around the picture like that:

![05_spark_5spp](https://user-images.githubusercontent.com/89559223/219353353-37e06851-894b-4a84-87d5-54ca9ef46d4e.png)

As we add more spp to average this noise, the result gets even worse:

![06_spark_50spp_even_worse](https://user-images.githubusercontent.com/89559223/219353650-2d026b14-7981-4b9d-ab48-b07987b04254.png)

I finally found that problem is caused by a Numerical accuracy bug. It makes the scattered ray intersects to it's original shade point(self intersect) so the energy of a light start to add to a extremely large value. After solved this problem, we got this result without salt pepper noise:

![14_spark_problem_solved_50spp](https://user-images.githubusercontent.com/89559223/219354615-1157b174-80e0-4119-92e1-1f186643d8a0.png)



#### 2023/02/13


Last time, I've correctly using the rendering equation to generate a picture that only contains diffuse material. However, when it turns to render a frame contains mental and other materials, mistakes comes~

![material_bug_direct](https://user-images.githubusercontent.com/89559223/218373766-51809145-d731-4ed7-9ac6-cdd3b3690b88.png)


That is because I've not apply a correct BRDF term of mental or dielectric material. 


Cause the BTDF term of the dielectric material should be considered, we just save it for last. Let's firstly consider the mental material BRDF. 

![increasing](https://user-images.githubusercontent.com/89559223/218373783-4618bda9-5cc2-425a-b9dc-7e2f8ac95841.png)


![fuzz0 9_correct_50spp](https://user-images.githubusercontent.com/89559223/218373827-d6eed970-ab87-4ae2-a121-49bcdd033653.png)


![anoWired](https://user-images.githubusercontent.com/89559223/218373859-63933920-9cd1-478a-99b3-73ed8ef1e24b.png)



To merge the BTDF into the current Rendering Equation is definitely the hardest part from now on. 

Well, if we take the problem into two independent parts, pure reflect and pure refract, it becomes easier to figure out. 

Pure reflect term is easy to simulate, cause we can refer to the mental surface we've just done. Pure refract surface is a little bit hard(cost me a day or two to get a correct result). Finally, I got it.

![muchCloser](https://user-images.githubusercontent.com/89559223/218373876-2ebf3ea4-013f-4505-8ce9-e3c82867475d.png)



Merge the two parts together then we got the final result.

![got_it](https://user-images.githubusercontent.com/89559223/218373883-9cb29d40-9cff-4dfe-9b85-2facc791975e.png)


(Besides, there is still some bugs in my code, it seems that the energy distribution is not true and the refract part is always too strong for the dielectric surface.)

#### 2023/02/07

Finally, the program can correctlly sample the light soure instead of randomly scatter rays into the world space. Here's some problems that I've met.

Firstly, I turn to test simpler sphere geometry rather than confused "complex" bunnny object. And I got that(wrong pixel is still marked with color green)

![weirdGhostGreen](https://user-images.githubusercontent.com/89559223/217293391-caf0ada5-95e8-425b-b4a2-160907657ad7.png)


Debugged with GDB, I've found that when the radiance value is too small, converting it from float to int(32 bit signed) may got wrong result: it turn to be the minimum value of 32 bit int, -2147483648 that is. After I've solved this, I got this pic as follow:

![direct_cast_ray_1spp_still_bug](https://user-images.githubusercontent.com/89559223/217293452-2c766290-1df1-40e4-b368-198263c9e77a.png)


A Little bit noisy. More spp always works.  

![direct_cast_ray_5spp_still_bug](https://user-images.githubusercontent.com/89559223/217293497-c2c1756d-eab4-407a-a30c-36a403430c37.png)


Wait, what's going on?! It supposed to be much more lighter in the conner of those spheres, however, it becomes dimmer! After checking the code, I've found tha I've fogot to initialize the **RussianRoulette** value, cause **L_indir** use it as a divisor, it always gets a "Nan" value. After fix this bug, I got the follow pic:

512*512 20spp using 8.9seconds

![direct_cast_ray_20spp_8 9s](https://user-images.githubusercontent.com/89559223/217293623-95b2fc5c-1cdd-418a-ad38-bfdd354575e5.png)


Comparing with Whitted styled rendering result.

512*512 100spp using 26.6seconds(same rendering quality)

![incorrect_whitted_styled_100spp_26 6s](https://user-images.githubusercontent.com/89559223/217293699-ab65fd4f-f2cd-4ba2-9b16-e794556c3a26.png)



512*512 20spp using 5.2seconds(same light source power) In this pic, we can see that Whitted styled ray tracing result is WRONG! It cannot present the true light transform law. What is correct? Rendering Equation of course.

![incorrect_whitted_styled_20spp_5 2s](https://user-images.githubusercontent.com/89559223/217294363-c7028d59-c409-4ad9-ae4a-866ca5e6f901.png)


#### 2023/02/06

I tried directly sampling from light source instead of scattering light all randomly. However, currently the render result is not correct. The incorrect shading point is marked with green color. I thought the wrong result may caused by too less "spp" but it turn worse when I add more rays cast into single pixel.

By the way, the correct color of this bunny should be yellow(Actually, you can still find some correctly rendered point in it)!

Here's the result of 1spp, green point marks the wrong shading result.

![sampleLight1spp](https://user-images.githubusercontent.com/89559223/216896083-fd1a70cf-90ca-43ec-938e-114b8dcabe91.png)

Clearer when using 5spp:

![sampleLight5spp](https://user-images.githubusercontent.com/89559223/216896160-26d42695-96ef-4880-a872-403ac65fb496.png)




#### 2023/02/04

Actually, we have indicated that there is something wrong when we use bvh accel constructure to render a frame, you may have already found that there is "black noisy point" in the object. However, it does not appear when we use naive render method. Today, we've found that is caused by a little bug: I've always been using "std::numeric_limits<float>::min()" as the ray transformation time's minimum limit. 

However, it's not true, cause the secondary scattered ray may intersect to the surface who generateit, that is to say, ray's hit point is the ray's origin position... In that case, the rec.t parameter can be extremely small but still larger than "std::numeric_limits<float>::min()", which makes the ray iteratively bounce and intersect to itself and retrun an vec(0,0,0) as a black noisy shading point.

Currently, this bug has been solved. Let's look some of the comparation result.


![bvh_tree_result_mental](https://user-images.githubusercontent.com/89559223/216764760-7fb679f1-c168-4d0f-9531-e9d6e654ed84.png)
 
This is the uncorrect result rendered by original bvh accel constructure.

 
 
![correct_naive_result_mental](https://user-images.githubusercontent.com/89559223/216764807-fd24fed6-49c6-4db7-a39f-22b1dceaa851.png)
 
"Ground truth~! with naive render method"

 
 
![bvh_changed_1spp_mental](https://user-images.githubusercontent.com/89559223/216764872-9ae99108-4b48-46ab-9778-0d71eb9dc7bc.png)
 
After fixing the bug, using bvh accel constructure, minimum transformation time is limited by 0.001, you can compare this with the upper pic.


 
![bvh](https://user-images.githubusercontent.com/89559223/216764972-2fbcb8be-7e09-44c1-ab9f-333c8250feaf.png)
 
Also, complex scene is rendered correctlly and you can still compare this pic with the 2023/01/08 version. No much black noisy shading point exist.
 
 
 
#### 2023/01/08

We optimized the class hierarchy as mentioned before. However, after I added the middle abstract level class "primitive" between class "hitable" and class "triangle", strange things happened: although I've changed the bvh constructure at the same time, the accelerate efficiency seems slow down apparently, even much slower than the brute-force method! Besides, still have different visual effect although I'm using the same shading method. Apparently, there is some bugs which slow down the program I've not found. The rendering result as follows: 

Right one using burte force method using 15 seconds. Left one with bvh constructure cost 24 seconds. (200*200 resolution; 1spp; 1500 primitives). The bvh tree is constructed inside the model, but much slower when rendering.

![bvh_bunny_200_200_1spp_1500prims_24s](https://user-images.githubusercontent.com/89559223/211190277-c81d9548-cf27-4a27-a7e2-b28d0e47d0f7.png)
![naive_bunny_200_200_1spp_1500prims_15s](https://user-images.githubusercontent.com/89559223/211190280-2d08d399-b6fb-4502-8a8c-a1c12f0a9c8f.png)

Bottom: burte force(20 seconds); Top: bvh(7 seconds).(512*512 resolution; 1spp; 900 spheres) However, this bvh tree is constructed on the scene but not inner models which means each sphere is regarded as a single object(model). Under this circumstance, bvh acceleration behaves its advantage.

![bvh_scene_512_512_1spp_900spheres_7s](https://user-images.githubusercontent.com/89559223/211190412-55104476-fb24-4d79-899e-a46cbe633ee0.png)
![naive_scene_512_512_1spp_900spheres_20s](https://user-images.githubusercontent.com/89559223/211190417-c9c46d6e-b7a0-4c61-8ec8-fe30c7b62de8.png)

 
 
#### 2023/01/06

We got global acceleration this time and firstly we can test a typical scene in the e-book <Ray tracing in one weekend>.

with acceleration(30s) & without acceleration(366s) but there is still something wrong in the rendered result(I doubt why the same scene can have that much differences in illumination, so there must be some bugs). 

![3200balls_512_512_5spp_30seconds](https://user-images.githubusercontent.com/89559223/211009785-19de6d16-c687-4284-b757-bc05efa56f81.png)

![3200balls_without_accel_512_512_5spp_366seconds](https://user-images.githubusercontent.com/89559223/211009840-7e6f7dee-6870-4a50-95ac-06e239655ada.png)



Then, we test multi-sphere in the scene and a triangle list with 500 primitives at the same time. Noted that the triangle list is regarded as a group, so it use inner-model-acceleration we've done last day. Besides, we use another much brighter skybox. Although enforced by acceleration constructure, it still cost me 10mins to render.

![512_512_900sphere_500tris_10spp_593seconds](https://user-images.githubusercontent.com/89559223/211009864-b053d0fe-a51b-4235-a3f4-6f6b3dad996c.png)

 
 
#### 2023/01/05

Firstly, we randomly generate multi-triangles（100 or so） and rendered them by brute-force traversal algorithm. Then test our bvh_tree accelerate constructure. (7.8seconds for the Former while less than a second after acceleration).

![Multi_Tris](https://user-images.githubusercontent.com/89559223/210777354-a56dd5e6-fa51-4239-afdc-98092e9f2fb2.png)

Then we try the bunny with lower resolution(1500 primitives or so).

![MirrorBunny_1k_prims_400_400_50spp_95s](https://user-images.githubusercontent.com/89559223/210777662-feb22f13-dca0-4edd-a414-0d169c0814ad.png)
![bunny_flow0](https://user-images.githubusercontent.com/89559223/210777678-4f07d7b7-f8fb-4180-ae91-3182f01c9826.png)

 
 
 #### 2023/01/03

ImageTexture with noisy ground.

![ImageTexture](https://user-images.githubusercontent.com/89559223/210772856-bf1198d4-cfc5-4d89-b158-1435f9c02836.png)

Sky Box is added！And we put a pure mirror sphere in the middle of our view cone.

![SkyBox](https://user-images.githubusercontent.com/89559223/210775943-c9920584-2fb9-4153-84d7-cffa81dc799e.png)



 

#### 2022/12/31

Multi-Triangles.

![TestTriangleList](https://user-images.githubusercontent.com/89559223/210775026-aa8aa0b0-a13a-4334-9c67-9079cb277ca6.png)


Classic Cornell Box (Model imported √ not rendered by generating basic geometry)（uhhhhh~ So noisy so far~ but still cost few seconds）.

![CornellBox](https://user-images.githubusercontent.com/89559223/210775353-964c1763-4c67-473a-9f6a-44d76692809b.png)


 
 
#### 2022/12/30

Basic Scene with basic geometry.

![SampleRGB](https://user-images.githubusercontent.com/89559223/210774486-e8228452-9658-4982-acff-a6ba477c5fd3.png)


Single Triangle.

![TestTriangle](https://user-images.githubusercontent.com/89559223/210774838-6456d714-fc19-4bf3-98b6-5f17a277b922.png)





