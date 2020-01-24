USE dsmith_04_opencl;

# What are the LLVM testbeds?

SELECT testbeds.id AS testbed_id,
       optimizations,
       platform,
       driver
FROM testbeds
LEFT JOIN platforms ON testbeds.platform_id=platforms.id
WHERE platform = 'clang';

-- Results:
-- '31','1','clang','3.6.2'
-- '32','1','clang','3.7.1'
-- '33','1','clang','3.8.1'
-- '34','1','clang','3.9.1'
-- '35','1','clang','4.0.1'
-- '36','1','clang','5.0.0'
-- '37','1','clang','6.0.0'
-- '38','1','clang','trunk'

SELECT results.id,
       assertions.assertion,
       results.outcome,
       programs.src
FROM results
LEFT JOIN testbeds ON results.testbed_id = testbeds.id
LEFT JOIN platforms ON testbeds.platform_id = platforms.id
LEFT JOIN testcases ON results.testcase_id = testcases.id
LEFT JOIN programs ON testcases.program_id = programs.id
LEFT JOIN stderrs ON results.stderr_id = stderrs.id
LEFT JOIN assertions ON stderrs.assertion_id = assertions.id
WHERE results.id >= 0
  AND testbeds.id =
    (SELECT testbeds.id
     FROM testbeds
     LEFT JOIN platforms ON testbeds.platform_id=platforms.id
     WHERE platform = 'clang'
       AND driver = '3.6.2' )
ORDER BY results.id
LIMIT 100;
