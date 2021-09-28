using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

public static class RuntimeTextureCompression
{

    private static ComputeShader shader;
    private static GraphicsFormat graphicsFormat;

    public static void Init()
    {
#if !UNITY_EDITOR
        shader = AssetBundleManager.Instance.LoadAssetSync<ComputeShader>("ComputeShader/Compress.compute");
#else
        shader = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/ComputeShader/Compress.compute");
#endif
    }

    //Compress RGBA32 Texture to ETC2, only square textures are supported
    public static Texture2D Compress(Texture sourceTex, bool isAlpha)
    {
        if (!SystemInfo.supportsComputeShaders || SystemInfo.copyTextureSupport == CopyTextureSupport.None)
        {
            return null;
        }
#if (UNITY_ANDROID || UNITY_IOS) && !UNITY_EDITOR
        if (isAlpha){
            graphicsFormat = GraphicsFormat.RGBA_ETC2_UNorm;
        }
        else{
            graphicsFormat = GraphicsFormat.RGBA_ASTC4X4_UNorm;
        }
#else
        return null;
#endif
        if (shader == null){
            Init();
            Debug.Log("Shader Name " + shader.name);
        }
        int width = sourceTex.width;
        int height = sourceTex.height;
        int compress_width = width / 4;
        int compress_height = height / 4;
        int mipmapCount = sourceTex.mipmapCount;
        int[] DestRect = new int[4] { 0, 0, width, height };
        int kernelHandle;

        RenderTexture tex = RenderTexture.GetTemporary(compress_width, compress_height, 0, GraphicsFormat.R32G32B32A32_UInt);
        tex.enableRandomWrite = true;
        tex.useMipMap = true;
        tex.autoGenerateMips = false;
        tex.Create();
        mipmapCount = Mathf.Min(mipmapCount, tex.mipmapCount);

        Texture2D copyTex = new Texture2D(width, height, graphicsFormat, mipmapCount, TextureCreationFlags.MipChain);
        copyTex.name = sourceTex.name;
        copyTex.anisoLevel = 6;

        UnityEngine.Profiling.Profiler.BeginSample("Compress Texture");
        for (int i = 0; i < mipmapCount; i++)
        {
            if (i > 0)
            {
                DestRect[2] = DestRect[2] >> 1;
                DestRect[3] = DestRect[3] >> 1;
            }
            shader.SetInts("DestRect", DestRect);
            int compressedMipMapSize = compress_width >> i;
            if (compressedMipMapSize >= 8)
            {
#if (UNITY_ANDROID || UNITY_IOS) && !UNITY_EDITOR
                if (isAlpha){
                    kernelHandle = shader.FindKernel("CSMainETC2_8");
                }
                else{
                    kernelHandle = shader.FindKernel("CSMainASTC_8");
                }
#else
                kernelHandle = shader.FindKernel("CSMainBC3_8");
#endif
                shader.SetTexture(kernelHandle, "RenderTexture0", sourceTex, i);
                shader.SetTexture(kernelHandle, "Result", tex, i);
                shader.Dispatch(kernelHandle, compressedMipMapSize / 8, compressedMipMapSize / 8, 1);
            }
            else if (compressedMipMapSize >= 4)
            {
#if (UNITY_ANDROID || UNITY_IOS) && !UNITY_EDITOR
                if (isAlpha){
                    kernelHandle = shader.FindKernel("CSMainETC2_4");
                }
                else{
                    kernelHandle = shader.FindKernel("CSMainASTC_4");
                }
#else
                kernelHandle = shader.FindKernel("CSMainBC3_4");
#endif
                shader.SetTexture(kernelHandle, "RenderTexture0", sourceTex, i);
                shader.SetTexture(kernelHandle, "Result", tex, i);
                shader.Dispatch(kernelHandle, compressedMipMapSize / 4, compressedMipMapSize / 4, 1);
            }
            else if (compressedMipMapSize >= 2)
            {
#if (UNITY_ANDROID || UNITY_IOS) && !UNITY_EDITOR
                if (isAlpha){
                    kernelHandle = shader.FindKernel("CSMainETC2_2");
                }
                else{
                    kernelHandle = shader.FindKernel("CSMainASTC_2");
                }
#else
                kernelHandle = shader.FindKernel("CSMainBC3_2");
#endif
                shader.SetTexture(kernelHandle, "RenderTexture0", sourceTex, i);
                shader.SetTexture(kernelHandle, "Result", tex, i);
                shader.Dispatch(kernelHandle, compressedMipMapSize / 2, compressedMipMapSize / 2, 1);
            }
            else
            {
#if (UNITY_ANDROID || UNITY_IOS) && !UNITY_EDITOR
                if (isAlpha){
                    kernelHandle = shader.FindKernel("CSMainETC2_1");
                }
                else{
                    kernelHandle = shader.FindKernel("CSMainASTC_1");
                }
#else
                kernelHandle = shader.FindKernel("CSMainBC3_1");
#endif
                shader.SetTexture(kernelHandle, "RenderTexture0", sourceTex, i);
                shader.SetTexture(kernelHandle, "Result", tex, i);
                shader.Dispatch(kernelHandle, compressedMipMapSize, compressedMipMapSize, 1);
            }
            Graphics.CopyTexture(tex, 0, i, 0, 0, compress_width >> i, compress_height >> i, copyTex, 0, i, 0, 0);
        }
        UnityEngine.Profiling.Profiler.EndSample();
        RenderTexture.ReleaseTemporary(tex);
        return copyTex;
    }
}
