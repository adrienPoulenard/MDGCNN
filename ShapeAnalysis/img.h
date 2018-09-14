#pragma once
#ifndef IMG_H
#define IMG_H
//#include <stb_image.h>
//#include <stb_image_write.h>

/*bool write_image_jpg(const std::vector<uint8_t>& img, int w, int h,
	const std::string& path, const std::string& img_name) {
	std::string img_path = path + "/" + img_name + ".jpg";
	stbi_write_jpg(img_path.c_str(), w, h, 3, &img[0], 100);
}

bool write_image_png(const std::vector<uint8_t>& img, int w, int h,
	const std::string& path, const std::string& img_name) {

}*/

bool write_image_ppm(const std::vector<uint8_t>& img, int w, int h,
	const std::string& path, const std::string& img_name) {
	const int dimx = w, dimy = h;
	std::string path_ = path + "/" + img_name + ".ppm";
	int i, j;
	FILE *fp = fopen(path_.c_str(), "wb"); /* b - binary mode */
	(void)fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
	for (j = 0; j < dimy; ++j)
	{
		for (i = 0; i < dimx; ++i)
		{
			static unsigned char color[3];
			
			//color[0] = img[w*h * 0 + (i*h + j)];  /* red */
			//color[1] = img[w*h * 1 + (i*h + j)];  /* green */
			//color[2] = img[w*h * 2 + (i*h + j)];  /* blue */

			color[0] = img[w*h * 0 + (j*h + i)];  /* red */
			color[1] = img[w*h * 1 + (j*h + i)];  /* green */
			color[2] = img[w*h * 2 + (j*h + i)];  /* blue */
			(void)fwrite(color, 1, 3, fp);
		}
	}
	(void)fclose(fp);
	return true;
}

void test_img(const std::string& path) {
	std::vector<uint8_t> img(3 * 32 * 32);
	for (int i = 0; i < img.size(); i++) {
		img[i] = 55;
	}
	write_image_ppm(img, 32, 32, path, "test");
}

#endif IMG_H