#include "config.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
//#define _CRT_SECURE_NO_WARNINGS

// VCでのリークチェック（_CrtSetDbgFlagも有効に）
//#define _CRTDBG_MAP_ALLOC #include <stdlib.h> #include <crtdbg.h>
// Visual Leak Detector でのチェック
// #include <vld.h>

#include <iostream>
#include <fstream>
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


void save(FB<ByteColor> *src, const char *filename)
{
	int w = src->getWidth();
	int h = src->getHeight();
	int comp = STBI_rgb; // RGB
	int stride_in_bytes = 3 * w;
	int result = stbi_write_png(filename, w, h, comp, src->ref(0), stride_in_bytes);
}

int main()
{
//	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF | _CRTDBG_LEAK_CHECK_DF);
	time_t t0 = time(NULL);
	time_t t_last = 0;
	int count = 0;
	int frame = 0;
	bool ret;
	int current;
	FrameBuffer *fb[3];
	renderer *pRenderer;
	HDRLoaderResult ibl_data;

	FB<ByteColor> *image = new FB<ByteColor>(WIDTH, HEIGHT);
	if (!image) goto image_failed;

	// frame buffer の初期化
	current = 0;
	fb[0] = new FrameBuffer(WIDTH, HEIGHT);// ダブルバッファ1
	if (!fb[0])goto fb0_failed;
	fb[1] = new FrameBuffer(WIDTH, HEIGHT);// ダブルバッファ2
	if (!fb[1])goto fb1_failed;
	fb[2] = new FrameBuffer(WIDTH, HEIGHT);// 法線マップ用
	if (!fb[2])goto fb2_failed;


	FB<double> *FB_Lum[2];
	FB_Lum[0] = new FB<double>(WIDTH, HEIGHT);
	FB_Lum[1] = new FB<double>(WIDTH, HEIGHT);

	pRenderer = new renderer(WIDTH, HEIGHT);
	if (!pRenderer)goto renderer_failed;

	ret = HDRLoader::load("media/Tokyo_BigSight/Tokyo_BigSight_3k.hdr", ibl_data);
//	ret = HDRLoader::load("media/Ridgecrest_Road/Ridgecrest_Road_Env.hdr", ibl_data);

	pRenderer->setIBL(ibl_data.width, ibl_data.height, ibl_data.cols);
	SAFE_DELETE_ARRAY(ibl_data.cols);// コピーされるので、実体は使われない

	// 初期描画
	pRenderer->update(fb[1 - current], fb[current], fb[2]);
	fb[current]->resolve(image);
	save(image, "1st_render.png");
	current = 1 - current;

	// メディアンフィルタでフィルタリング
	pRenderer->median_filter(*fb[1 - current], *fb[current]);
	fb[current]->resolve(image);
	save(image, "median.png");
	current = 1 - current;

	// 輝度抽出検出
	pRenderer->get_luminance(*fb[1 - current], *FB_Lum[0]);
	FB_Lum[0]->resolve(image);
	save(image, "luminance.png");
	current = 1 - current;

	// エッジ検出
	pRenderer->edge_detection(*FB_Lum[0], *FB_Lum[1]);
	FB_Lum[1]->resolve(image);
	save(image, "edge.png");
	current = 1 - current;

	// エッジのガウスブラー
	pRenderer->gauss_blur_x(*FB_Lum[1], *FB_Lum[0]);
	pRenderer->gauss_blur_y(*FB_Lum[0], *FB_Lum[1]);
	FB_Lum[1]->resolve(image);
	save(image, "edge_blurred.png");

	// 法線方向の検出
	pRenderer->compute_normal(*FB_Lum[1], *fb[2]);
	fb[2]->resolve(image);
	save(image, "normal.png");

	// 再初期化
	frame = 0;
	fb[current]->clear();
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
			c -= 1; // 000.pngからはじまる
			char filename[256] = { '0', '0', '0', '.', 'p', 'n', 'g', '\0' };
			filename[1] = '0' + (c / 10);
			filename[2] = '0' + (c % 10);
			fb[current]->resolve(image, 1.0 / (double)frame);
			save(image, filename);
			count++;
		}

		if (finished) break;

		// swap
		current = 1 - current;
		t_last = t;
	}while (true);

	SAFE_DELETE(pRenderer);
renderer_failed:

	SAFE_DELETE(FB_Lum[1]);
	SAFE_DELETE(FB_Lum[0]);

	SAFE_DELETE(fb[2]);
fb2_failed:
	SAFE_DELETE(fb[1]);
fb1_failed:
	SAFE_DELETE(fb[0]);
fb0_failed:
	SAFE_DELETE(image);
image_failed:

	// log 出力
	std::ofstream f;
	f.open("log.txt", std::ios::out);
	f << "frame count:" << frame << "<br>" << std::endl;
	
	return 0;
}

