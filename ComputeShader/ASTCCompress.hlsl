/*
Fast simple ASTC4x4 Compression for mobile, only single partition and single plane supported
MemoryLayout: Total 128bit for 4X4 texels
 -------------------------------------------------------------
|                  57bits  Weights                             |
 -------------------------------------------------------------
|                                                             |
 -------------------------------------------------------------
|                |       48bits EndPoints Color               |
 -------------------------------------------------------------
|                  48bits EndPoints Color             | 4bits |
 -------------------------------------------------------------
| CEM          |2bits Part|      11bits BlockMode             |
 -------------------------------------------------------------
 Note that the weights data is laid from high bits to low bits. 

*/

static const uint weight_table[12] = { 0, 4, 8, 2, 6, 10, 11, 7, 3, 9, 5, 1 };

void TranferToUNorm8(in float3 texels[16], out uint3 texelsUNorm8[16]) {
	for (int i = 0; i < 16; i++) {
		texelsUNorm8[i] = uint3(round(texels[i] * 255));
	}
}

bool IsConstantColor(in float3 blockBaseColor[16], out float3 color) {
    color = blockBaseColor[0];
    float sum = 0;
    float3 difference;
	for (int i = 1; i < 16; i++) {
        difference = blockBaseColor[i] - color;     
        sum += dot(difference, difference);
	}
    if (sum > 0.0001f) {
        return false;
    }
	return true;
}

uint4 ConstantBlockToPhysical(float3 color) {
    
	uint4 color_16bit = uint4(0, 0, 0, 0);
	color_16bit.x = (uint)(color.x * 65535);
	color_16bit.y = (uint)(color.y * 65535);
	color_16bit.z = (uint)(color.z * 65535);
	color_16bit.w = 0;
    
	uint z = 0; 
	z = z | color_16bit.x;
	z = z | (color_16bit.y << 16);

	uint w = 0;
	w = w | color_16bit.z;
	
	return uint4(4294966780, 4294967295, z, w);
}

float3 GetMean(in uint3 texels[16]) {
	float3 sum = float3(0, 0, 0);
	for (int i = 0; i < 16; i++) {
		sum = sum + texels[i];
	}
	return sum / 16;
}

float GetComponent(float3 texel, int index) {
	if (index == 0) {
		return texel.x;
	}
	else if (index == 1) {
		return texel.y;
	}
	else {
		return texel.z;
	}
}

float3x3 Covariance(in float3 texels[16]) {
	float3x3 cov;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			float s = 0;
			for (int k = 0; k < 16; k++) {
				s += GetComponent(texels[k], i) * GetComponent(texels[k], j);
			}
			cov[i][j] = s / 15.0;
		}
	}
	return cov;
}

float3 SafeNormalize(float3 v) {
    if (length(v) == 0) {
        return float3(0.57736f, 0.57736f, 0.57736f);
    }
    else {
        return normalize(v);
    }
}

float3 EigenVector(in float3x3 cov) {
	float3 b = normalize(float3(1, 3, 2));
	for (int i = 0; i < 8; i++) {
		b = SafeNormalize(mul(cov, b));
	}
	return b;
}

void PCA(in uint3 texels[16], out float3 k, out float3 m) {
	m = GetMean(texels);
	float3 texelsFloat[16];
	for (int i = 0; i < 16; i++) {
		texelsFloat[i] = texels[i] - m;
	}
	float3x3 cov = Covariance(texelsFloat);
	k = EigenVector(cov);
}

void FindMinMax(in uint3 texels[16], float3 k, float3 m, out float3 e0, out float3 e1) {
	float t = dot(texels[0] - m, k);
	float min_t = t;
	float max_t = t;

	for (int i = 1; i < 16; i++) {
		t = dot(texels[i] - m, k);
		min_t = min(min_t, t);
		max_t = max(max_t, t);
	}

	e0 = k * min_t + m;
	e1 = k * max_t + m;

	e0 = clamp(e0, 0, 255);
	e1 = clamp(e1, 0, 255);
}

void PrepareEndPointsForISE(in uint3 endpoint_quantized[2], out uint flat_endpoints[6]) {
    flat_endpoints[0] = endpoint_quantized[0].x;
    flat_endpoints[1] = endpoint_quantized[1].x;
    flat_endpoints[2] = endpoint_quantized[0].y;
    flat_endpoints[3] = endpoint_quantized[1].y;
    flat_endpoints[4] = endpoint_quantized[0].z;
    flat_endpoints[5] = endpoint_quantized[1].z;
}

void EncodeRGBDirect(float3 e0, float3 e1, out uint3 endpoint_unquantized[2], out uint3 endpoint_quantized[2]) {
    uint3 rounded_e0 = round(e0);
    uint3 rounded_e1 = round(e1);

    if ((rounded_e0.r + rounded_e0.g + rounded_e0.b) > (rounded_e1.r + rounded_e1.g + rounded_e1.b)) {
        endpoint_unquantized[0] = rounded_e1;
        endpoint_unquantized[1] = rounded_e0;
    }
    else {
        endpoint_unquantized[0] = rounded_e0;
        endpoint_unquantized[1] = rounded_e1;
    }

    endpoint_quantized[0] = endpoint_unquantized[0];
    endpoint_quantized[1] = endpoint_unquantized[1];
}

void EncodeQuantizedWeightsRGB(in uint3 texels[16], uint3 e0, uint3 e1, out uint weight_quantized[16]) {
    float3 k = float3(e1)-float3(e0);
    float3 m = float3(e0);

    float scale = (e1.x - e0.x) + (e1.y - e0.y) + (e1.z - e0.z);

    float kk = dot(k, k);
    kk = max(1, kk);
    for (int i = 0; i < 16; i++) {
        float3 difference = texels[i] - m;
        
        float weights = dot(difference, k) * 11.f / kk * min(scale, 1.f);
        float projected_weight = clamp(weights, 0.f, 11.f);
        
        weight_quantized[i] = weight_table[uint(projected_weight)];
    }
}

void SplitHighLow(uint n, uint i, out uint high, out uint low) {
    uint low_mask = (1 << i) - 1;

    low = n & low_mask;
    high = n >> i;
}

void IntegerSequenceEncodeEndPoints(in uint numbers[6], out int output[6]) {
    for (int i = 0; i < 6; i++) {
        output[i] = numbers[i];
    }
}

uint GetBits(uint number, int count, int lsb) {
    float result = (float)number / (float)(pow(2, lsb));
    return (uint)(result) & ((1 << count) - 1);
    //Weird bug, the following code can lead overflow on Adreno GPUs. For example, number = 2, lsb = 1, number >> lsb = 4294967295
    //return (number >> lsb) & ((1 << count) - 1);
}

void IntegerSequenceEncodeWeights(in uint numbers[16], out int output[24]) {
    int j = 0;
    for (int i = 0; i < 15; i += 5) {
        uint b0 = numbers[i];
        uint b1 = numbers[i + 1];
        uint b2 = numbers[i + 2];
        uint b3 = numbers[i + 3];
        uint b4 = numbers[i + 4];

        uint t0, t1, t2, t3, t4;
        uint m0, m1, m2, m3, m4;

        SplitHighLow(b0, 2, t0, m0);
        SplitHighLow(b1, 2, t1, m1);
        SplitHighLow(b2, 2, t2, m2);
        SplitHighLow(b3, 2, t3, m3);
        SplitHighLow(b4, 2, t4, m4);

        
        uint c = 0;
        if (t1 == 2 && t2 == 2) {
            c = 12 + t0;
        }
        else if (t2 == 2) {
            c = t1 * 16 + t0 * 4 + 3;
        }
        else {
            c = t2 * 16 + t1 * 4 + t0;
        }

        uint t = 0;
        if (t3 == 2 && t4 == 2) {
            t = GetBits(c, 3, 2) * 32 + 28 + GetBits(c, 2, 0);
        }
        else {
            t = GetBits(c, 5, 0);
            if (t4 == 2) {
                t += t3 * 128 + 96;
            }
            else {
                t += t4 * 128 + t3 * 32;
            }
        }
        

        uint packed = t;

        output[j] = m0;
        output[j + 1] = m1;
        output[j + 2] = m2;
        output[j + 3] = m3;
        output[j + 4] = m4;
        output[j + 5] = packed;

        j += 6;
    }
    uint t0;
    uint m0;
    SplitHighLow(numbers[15], 2, t0, m0);
    uint packed = GetBits(t0, 5, 0);
    output[18] = m0;
    output[19] = 0;
    output[20] = 0;
    output[21] = 0;
    output[22] = 0;
    output[23] = packed;

}

uint GetBit(uint number, int index) {
    return (number >> index) & 1;
}


uint ReverseBits(uint input)
{
    uint t = input;
    t = (t << 16) | (t >> 16);
    t = ((t & 16711935) << 8) | ((t & 4278255360) >> 8);
    t = ((t & 252645135) << 4) | ((t & 4042322160) >> 4);
    t = ((t & 858993459) << 2) | ((t & 3435973836) >> 2);
    t = ((t & 1431655765) << 1) | ((t & 2863311530) >> 1);

    return t;
}

void SetWeights(in int input[24], inout uint z, inout uint w) {

    uint x = 0;
    uint y = 0;
    x |= input[0];
    x |= GetBits(input[5], 2, 0) << 2;
    x |= input[1] << 4;
    x |= GetBits(input[5], 2, 2) << 6;
    x |= input[2] << 8;
    x |= GetBits(input[5], 1, 4) << 10;
    x |= input[3] << 11;
    x |= GetBits(input[5], 2, 5) << 13;
    x |= input[4] << 15;
    x |= GetBits(input[5], 1, 7) << 17;

    x |= input[6] << 18;
    x |= GetBits(input[11], 2, 0) << 20;
    x |= input[7] << 22;
    x |= GetBits(input[11], 2, 2) << 24;
    x |= input[8] << 26;
    x |= GetBits(input[11], 1, 4) << 28;
    x |= input[9] << 29;
    uint temp = GetBits(input[11], 2, 5);
    x |= GetBits(temp, 1, 0) << 31;

    y |= GetBits(temp, 1, 1);
    y |= input[10] << 1;
    y |= GetBits(input[11], 1, 7) << 3;
    y |= input[12] << 4;
    y |= GetBits(input[17], 2, 0) << 6;
    y |= input[13] << 8;
    y |= GetBits(input[17], 2, 2) << 10;
    y |= input[14] << 12;
    y |= GetBits(input[17], 1, 4) << 14;
    y |= input[15] << 15;
    y |= GetBits(input[17], 2, 5) << 17;
    y |= input[16] << 19;
    y |= GetBits(input[17], 1, 7) << 21;
    y |= input[18] << 22;
    y |= GetBits(input[23], 2, 0) << 24;


    w = w | ReverseBits(x);
    z = z | ReverseBits(y);
}


uint4 SymbolicToPhysical(in int endpointISE[6], in int weightISE[24]) {

    uint x = 66129;

    uint y = 0;
    uint z = 0;
    uint w = 0;

    x |= endpointISE[0] << 17;
    uint temp = GetBits(endpointISE[1], 7, 0);
    x |= temp << 25;
    temp = GetBits(endpointISE[1], 1, 7);
    y |= temp;
    y |= endpointISE[2] << 1;
    y |= endpointISE[3] << 9;
    y |= endpointISE[4] << 17;
    temp = GetBits(endpointISE[5], 7, 0);
    y |= temp << 25;
    temp = GetBits(endpointISE[5], 1, 7);
    z |= temp;

    
    SetWeights(weightISE, z, w);
    return uint4(x, y, z, w);
}





uint4 EncodeRGBSinglePartition(in uint3 texels[16], float3 e0, float3 e1) {
    int partition_index = 0;
    int partition_count = 1;

    int weight_quant = 7;
    int endpoint_quant = 20;
    
    uint3 endpoint_unquantized[2];
    uint3 endpoint_quantized[2];
	EncodeRGBDirect(e0, e1, endpoint_unquantized, endpoint_quantized);

    
	uint weight_quantized[16];
    EncodeQuantizedWeightsRGB(texels, endpoint_unquantized[0], endpoint_unquantized[1], weight_quantized);
    
    uint flat_endpoints[6];
    PrepareEndPointsForISE(endpoint_quantized, flat_endpoints);
    int endpoint_ise[6];
    IntegerSequenceEncodeEndPoints(flat_endpoints, endpoint_ise);

    int weight_ise[24];
    IntegerSequenceEncodeWeights(weight_quantized, weight_ise);

    return SymbolicToPhysical(endpoint_ise, weight_ise);
    
}

uint4 CompressBlock(in float3 blockBaseColor[16]) {
    
    float3 baseColor;
    if (IsConstantColor(blockBaseColor, baseColor)) {
        return ConstantBlockToPhysical(baseColor);
    }
    
    uint3 texelsUNorm8[16];
    TranferToUNorm8(blockBaseColor, texelsUNorm8);
    float3 k;
    float3 m;
    PCA(texelsUNorm8, k, m);
    float3 e0;
    float3 e1;
    FindMinMax(texelsUNorm8, k, m, e0, e1);
    //return uint4(round(e1), 0);
    return EncodeRGBSinglePartition(texelsUNorm8, e0, e1);
    //return uint4(0, 0, 0, 0);
}
