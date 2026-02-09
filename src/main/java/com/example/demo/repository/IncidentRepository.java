package com.example.demo.repository;

import com.example.demo.domain.Incident;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface IncidentRepository extends JpaRepository<Incident, Long> {
    // 기본 저장(save), 조회(find) 기능이 자동으로 포함됩니다.
}