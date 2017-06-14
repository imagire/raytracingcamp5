#include "config.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
//#define _CRT_SECURE_NO_WARNINGS

// VCでのリークチェック（_CrtSetDbgFlagも有効に）
//#define _CRTDBG_MAP_ALLOC #include <stdlib.h> #include <crtdbg.h>
// Visual Leak Detector でのチェック
// #include <vld.h>

#include <iostream>
#include <thread>
#include <omp.h>

#include "../sdk/stb/stb_image.h"
#include "../sdk/stb/stb_image_write.h"

#include "hdrloader.h"
#include "renderer.h"


// おおよそ30秒毎に、レンダリングの途中経過をbmpかpngで連番(000.png, 001.png, ...) で出力してください。
// 4分33秒以内に自動で終了してください。

//#define WIDTH 200
//#define HEIGHT 100
#define WIDTH 1920
#define HEIGHT 1080

#define OUTPUT_INTERVAL 30
#define FINISH_TIME (4 * 60 + 33)
#define FINISH_MARGIN 2

void save(const double *data, unsigned char *buf, const char *filename, int steps)
{
	const double coeff = 1.0 / (0.3 * (double)steps);
	
	#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < 3 * WIDTH * HEIGHT; i++) {
//			double tmp = data[i] / (double)steps;// tone mapping
			double tmp = 1.0 - exp(-data[i] * coeff);// tone mapping
			buf[i] = (unsigned char)(pow(tmp, 1.0 / 2.2) * 255.999);// gamma correct
		}
	}

	// save
	int w = WIDTH;
	int h = HEIGHT;
	int comp = STBI_rgb; // RGB
	int stride_in_bytes = 3 * w;
	int result = stbi_write_png(filename, w, h, comp, buf, stride_in_bytes);
}

static void initFB(double *fb)
{
	#pragma omp parallel for
	for (int i = 0; i < 3 * WIDTH * HEIGHT; i++) {
		fb[i] = 0.0;
	}
}

int main()
{
//	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF | _CRTDBG_LEAK_CHECK_DF);

	time_t t0 = time(NULL);
	time_t t_last = 0;
	int count = 0;

	unsigned char *image = new unsigned char[3 * WIDTH * HEIGHT];

	// frame buffer の初期化
	int current = 0;
	double *fb[3];
	fb[0] = new double[3 * WIDTH * HEIGHT];// ダブルバッファ1
	fb[1] = new double[3 * WIDTH * HEIGHT];// ダブルバッファ2
	fb[2] = new double[3 * WIDTH * HEIGHT];// 法線マップ用
	initFB(fb[0]);
	initFB(fb[1]);
	initFB(fb[2]);

	renderer *pRenderer = new renderer(WIDTH, HEIGHT);

	HDRLoaderResult result;
	bool ret = HDRLoader::load("media/Tokyo_BigSight/Tokyo_BigSight_3k.hdr", result);

	// 初期描画
	pRenderer->update(fb[1 - current], fb[current], fb[2]);
	save(fb[current], image, "1st_render.png", 1);
	current = 1 - current;

	// メディアンフィルタでフィルタリング
	pRenderer->median_filter(fb[1 - current], fb[current]);
	save(fb[current], image, "median.png", 1);
	current = 1 - current;

	// 輝度抽出検出
	pRenderer->get_luminance(fb[1 - current], fb[2]);
	save(fb[2], image, "luminance.png", 1);
	current = 1 - current;

	// エッジ検出
	pRenderer->edge_detection(fb[2], fb[1 - current]);
	save(fb[1 - current], image, "edge.png", 1);
	current = 1 - current;

	// エッジのガウスブラー
	pRenderer->gauss_blur_x(fb[current],fb[2]);
	pRenderer->gauss_blur_y(fb[2],fb[current]);
	save(fb[current], image, "edge_blurred.png", 1);

	// 法線方向の検出
	pRenderer->compute_normal(fb[current], fb[2]);
	save(fb[2], image, "normal.png", 1);

	// 再初期化
	int frame = 0;
	initFB(fb[current]);
	current = 1 - current;

	do
	{
		// fb[1-current] を読み込んで fb[current]にレンダリング
		pRenderer->update(fb[1 - current], fb[current], fb[2]);
		frame++;

		// 4分33秒以内に終了なので、前のフレームを考えてオーバーしそうならば終了する
		time_t t = time(NULL) - t0;
		bool finished = (FINISH_TIME - FINISH_MARGIN <= t + (t - t_last));

		// 30 秒ごとか最後に出力
		int c = (int)(t / OUTPUT_INTERVAL);
		if (count < c || finished) {
			// sprintf_s を使いたくなくて、古典的な手法
			char filename[256] = { '0', '0', '.', 'p', 'n', 'g', '\0' };
			filename[0] = '0' + (c / 10);
			filename[1] = '0' + (c % 10);
			save(fb[current], image, filename, frame);
			count++;
		}

		if (finished) break;

		// swap
		current = 1 - current;
		t_last = t;
	}while (true);

	delete pRenderer;
	delete[] image;
	delete[] fb[2];
	delete[] fb[1];
	delete[] fb[0];

	return 0;
}

