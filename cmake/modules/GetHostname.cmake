function(GetHostname __hostnamevar)
  find_program(HOSTNAME_BIN hostname)
  if (HOSTNAME_BIN)
    execute_process(COMMAND ${HOSTNAME_BIN}
      OUTPUT_VARIABLE HOSTNAME
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  else ()
    set(HOSTNAME "<unavailable>")
  endif ()
  set(${__hostnamevar} ${HOSTNAME} PARENT_SCOPE)
endfunction()