package com.example.demo.controller;

import com.example.demo.service.AiService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.Map;

@RestController
@RequestMapping("/api/reports")
@RequiredArgsConstructor // [필수] aiService 주입을 담당합니다.
public class ReportController {

    private final AiService aiService; // final 필수

    @PostMapping("/submit")
    public ResponseEntity<String> submitReport(
            @RequestParam("content") String content,
            @RequestParam("file") MultipartFile file) {
        aiService.processUnifiedReport(content, file);
        return ResponseEntity.ok("신고 분석 및 저장이 완료되었습니다.");
    }

    @PostMapping("/analyze")
    public String analyzeReport(@RequestBody Map<String, String> request) {
        aiService.processReport(request.get("content")); // 에러 해결
        return "텍스트 분석이 완료되었습니다.";
    }
}