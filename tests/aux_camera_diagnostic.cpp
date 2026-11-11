#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "io/hikrobot/hikrobot.hpp"
#include "tools/exiter.hpp"

// Raw camera preview only. No detector, attitude input or actuator connection.
int main(int argc, char ** argv)
{
  const cv::String keys =
    "{help h | | Show usage}"
    "{@config | configs/sentry.yaml | Camera configuration}"
    "{csv | | Optional CSV output (new file)}";
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config = cli.get<std::string>(0);
  const auto csv_path = cli.get<std::string>("csv");
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }
  try {
    const auto yaml = YAML::LoadFile(config);
    const std::string roles[] = {"left", "right"};
    const std::string serials[] = {
      yaml["omni_left_camera_serial_number"].as<std::string>(),
      yaml["omni_right_camera_serial_number"].as<std::string>()};
    if (serials[0].empty() || serials[1].empty() || serials[0] == serials[1]) {
      throw std::runtime_error("Set two distinct, nonempty auxiliary camera serial numbers.");
    }
    const double exposure = yaml["omni_exposure_ms"]
                              ? yaml["omni_exposure_ms"].as<double>()
                              : yaml["exposure_ms"].as<double>();
    const double gain = yaml["omni_gain"] ? yaml["omni_gain"].as<double>()
                                          : yaml["gain"].as<double>();
    const float frame_rate = yaml["omni_frame_rate"] ? yaml["omni_frame_rate"].as<float>() : 60.0F;
    if (!std::isfinite(exposure) || exposure <= 0 || !std::isfinite(gain) || gain < 0 ||
        !std::isfinite(frame_rate) || frame_rate <= 0) {
      throw std::runtime_error("Invalid camera exposure, gain or requested frame rate.");
    }
    std::ofstream csv;
    if (!csv_path.empty()) {
      std::ifstream existing(csv_path);
      if (existing.good()) throw std::runtime_error("CSV already exists; choose a new path.");
      csv.open(csv_path);
      if (!csv) throw std::runtime_error("Cannot create CSV output.");
      csv << "role,frame,host_frame_ns,read_ns,interval_ms,age_ms,width,height\n";
    }

    tools::Exiter exiter;
    using Clock = std::chrono::steady_clock;
    std::unique_ptr<io::HikRobot> cameras[2];
    struct StopAllCameras
    {
      ~StopAllCameras() { io::HikRobot::request_stop_all(); }
    } stop_all_cameras;
    cv::Mat frames[2];
    Clock::time_point stamps[2]{};
    auto rate_start = Clock::now();
    auto last_flush = rate_start;
    unsigned long long counts[2]{}, rate_counts[2]{};
    double fps[2]{}, interval_ms[2]{};
    for (int i = 0; i < 2; ++i) {
      std::cout << roles[i] << " configured serial: " << serials[i]
                << " | requested FPS: " << frame_rate << '\n';
      cameras[i] = std::make_unique<io::HikRobot>(
        exposure, gain, yaml["vid_pid"].as<std::string>(), serials[i], frame_rate);
      cv::namedWindow(roles[i], cv::WINDOW_NORMAL);
      cv::resizeWindow(roles[i], 800, 640);
    }
    std::cout << "Cover one lens at a time to verify roles. ESC or Q exits.\n";
    while (!exiter.exit()) {
      for (int i = 0; i < 2; ++i) {
        cv::Mat frame;
        Clock::time_point stamp;
        if (cameras[i]->try_read(frame, stamp) && !frame.empty()) {
          const auto read_time = Clock::now();
          interval_ms[i] = counts[i] ? std::chrono::duration<double, std::milli>(stamp - stamps[i]).count() : 0;
          frames[i] = frame;
          stamps[i] = stamp;
          ++counts[i];
          ++rate_counts[i];
          if (csv.is_open()) {
            csv << roles[i] << ',' << counts[i] << ','
                << std::chrono::duration_cast<std::chrono::nanoseconds>(stamp.time_since_epoch()).count() << ','
                << std::chrono::duration_cast<std::chrono::nanoseconds>(read_time.time_since_epoch()).count() << ','
                << interval_ms[i] << ','
                << std::chrono::duration<double, std::milli>(read_time - stamp).count() << ','
                << frame.cols << ',' << frame.rows << '\n';
          }
        }
      }
      const auto now = Clock::now();
      const double period = std::chrono::duration<double>(now - rate_start).count();
      if (period >= 1.0) {
        for (int i = 0; i < 2; ++i) {
          fps[i] = rate_counts[i] / period;
          rate_counts[i] = 0;
        }
        rate_start = now;
      }
      if (csv.is_open() && now - last_flush >= std::chrono::seconds(1)) {
        csv.flush();
        if (!csv) throw std::runtime_error("CSV write failed.");
        last_flush = now;
      }
      for (int i = 0; i < 2; ++i) {
        cv::Mat canvas(640, 800, CV_8UC3, cv::Scalar(20, 20, 20));
        if (!frames[i].empty()) {
          const double scale = std::min(800.0 / frames[i].cols, 480.0 / frames[i].rows);
          cv::Mat preview;
          cv::resize(frames[i], preview, cv::Size(), scale, scale);
          preview.copyTo(canvas(cv::Rect((800 - preview.cols) / 2, 160, preview.cols, preview.rows)));
        }
        const double age = counts[i] ? std::chrono::duration<double, std::milli>(now - stamps[i]).count() : 0;
        const std::string status = !counts[i] ? "WAITING" : age > 1000 ? "STALE" : "RECEIVING";
        const std::string lines[] = {
          roles[i] + " configured SN: " + serials[i],
          fmt::format("{} | observed FPS {:.1f} | frames {}", status, fps[i], counts[i]),
          counts[i] ? fmt::format("Host age {:.1f} ms | interval {:.1f} ms | {}x{}", age, interval_ms[i], frames[i].cols, frames[i].rows) : "No frame received",
          "Cover lens to verify role. Q / ESC exits."};
        for (int line = 0; line < 4; ++line) {
          cv::putText(canvas, lines[line], {12, 28 + line * 35}, cv::FONT_HERSHEY_SIMPLEX,
                      0.55, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        }
        cv::imshow(roles[i], canvas);
      }
      const int key = cv::waitKey(5) & 0xff;
      if (key == 27 || key == 'q' || key == 'Q') break;
      if (cv::getWindowProperty("left", cv::WND_PROP_VISIBLE) < 1 ||
          cv::getWindowProperty("right", cv::WND_PROP_VISIBLE) < 1) break;
    }
    cv::destroyAllWindows();
  } catch (const std::exception & error) {
    std::cerr << "Camera diagnostic: " << error.what() << '\n';
    return 1;
  }
  return 0;
}