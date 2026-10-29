# Consideraciones éticas

## 1. Análisis de sesgos

**¿El dataset tiene sesgos demográficos, culturales o de otro tipo?** No es
posible medir sesgo demográfico directo (edad, género) porque esas variables
no están disponibles: `DIM_CLIENTE` no tiene fecha de nacimiento, y la
columna `sexo` se quitó del esquema final de 16 columnas (ver
`CONTEXTO_PROYECTO.md`, sección 6). Esto es, en parte, una ventaja (evita una
fuente obvia de discriminación directa), pero también significa que un sesgo
demográfico *indirecto* (por ejemplo, vía código postal de la bodega o canasta
de productos asociada a ciertos grupos) no puede descartarse ni confirmarse
con los datos actuales — queda declarado como limitación de medición, no como
ausencia de sesgo.

**Sesgo por diseño de cobertura:** el modelo solo cubre clientes con 4+
visitas/año (~26% de los clientes con actividad). El 69,7% restante (clientes
esporádicos) queda sistemáticamente fuera de cualquier lista de retención
generada por este sistema — no porque no importen, sino porque no hay
suficiente historial para estimar su ritmo propio.

**Análisis de sesgo formal entre segmentos (Fairlearn, 17/10/2026).** Usando
`fairlearn.metrics.MetricFrame` sobre las predicciones de test, con el umbral
operativo real del proyecto (top 10% de la lista por score), se midió
`selection_rate` (qué fracción de cada grupo entra a la lista de contacto) y
`recall` (qué fracción de los verdaderos desertores de cada grupo es
capturada):

| Segmento de frecuencia | % del grupo que entra a la lista (top 10% global) | % de desertores reales capturados |
|---|---|---|
| Baja (4-6 visitas/año) | 21,45% | 31,48% |
| Media (7-12 visitas/año) | 6,08% | 13,21% |
| Alta (13+ visitas/año) | 3,42% | 10,97% |

Diferencia máxima entre grupos: 18,0 p.p. en `selection_rate`, 20,5 p.p. en
`recall`. Razón mínima entre grupos: 0,16 en `selection_rate` (el segmento
alto recibe ~6 veces menos lugares en la lista que el bajo, en proporción a su
tamaño) y 0,35 en `recall`.

**¿Qué grupos podrían verse perjudicados?** Cuando se usa un **umbral global**
(un solo top-10% para toda la base, que es como normalmente se arma una
campaña con presupuesto fijo), el ranking favorece sistemáticamente a los
clientes de **baja frecuencia** — tienen tasa base de churn más alta, así que
dominan el tramo superior del score. Los clientes de **alta frecuencia**, que
son los más valiosos y leales del negocio, quedan relativamente sub-representados
en la lista: solo el 11% de sus desertores reales se capturaría con un
presupuesto de campaña de 10% global, contra el 31% en el segmento bajo. Esto
es importante porque el segmento alto es también donde el modelo discrimina
*mejor* en términos relativos (mayor lift, ver `results/reports/informe_evaluacion_final.md`) —
hay señal suficiente para priorizarlos bien, pero un umbral global no la
aprovecha.

Para `es_comprador_cronico`: diferencia de 7,4 p.p. en `selection_rate` y 11,2
p.p. en `recall`, con el grupo crónico recibiendo más lugares y mayor recall —
razonable dado que tiene una tasa base de churn más alta, pero vale la pena
vigilarlo junto con la limitación de la sección 3 (confusión entre fin de
tratamiento y fuga comercial).

Resultados completos en `results/metrics/fairness_metricframe.json`.

## 2. Equidad y fairness

**Estrategia de mitigación recomendada (no implementada aún):** usar un umbral
de selección **por segmento** (por ejemplo, top 10% dentro de cada segmento de
frecuencia) en vez de un umbral global único, de forma que el presupuesto de
contacto se reparta de forma más proporcional al riesgo real de cada grupo,
en vez de que un segmento domine la lista completa por tener una tasa base más
alta. Esto es exactamente lo que hace `fairlearn.postprocessing` (ajuste de
umbrales por grupo) y queda como siguiente paso recomendado antes de operar el
modelo en producción real.

Las métricas de desempeño (PR-AUC, precision@k, lift@k) se desglosan siempre
por segmento de frecuencia y por comprador crónico (regla no negociable #5),
precisamente para poder detectar diferencias como la de arriba.

## 3. Confusión entre deserción y fin de tratamiento (limitación central)

El diseño no distingue con certeza entre un cliente que **se fue a la
competencia** y uno cuyo tratamiento crónico simplemente terminó (alta
médica, cambio de medicación, fallecimiento). Las dos features pensadas para
esa distinción (`es_comprador_cronico`, `pct_gasto_cronico_12m`) tienen
importancia casi nula según SHAP (ver `arquitectura.md`) — el modelo **no**
resuelve esta limitación, solo la declara. Contactar a un cliente que dejó de
comprar por una razón médica ajena al negocio, ofreciéndole una promoción de
retención, puede ser percibido como insensible; esto debería considerarse
antes de automatizar el contacto sin revisión humana, especialmente en el
segmento de comprador crónico.

## 4. Privacidad

Ningún dato personal de `DIM_CLIENTE` (nombre, documento, teléfono, correo,
dirección) sale en ninguna extracción de análisis — regla no negociable #9
de `PLAN_DE_ACCION.md`. El modelo trabaja únicamente con `id_cliente` como
identificador y variables de comportamiento agregadas (recencia, frecuencia,
monto, canasta, antigüedad). La credencial de conexión a SQL Server es de
**solo lectura**: el proyecto no puede ni crear ni modificar nada en la base
de origen (regla no negociable #10). Nota adicional: las categorías de
compra en una farmacia pueden revelar condiciones de salud (vía
`producto_cronico`); el modelo agrega esto a nivel de `pct_gasto_cronico_12m`
sin exponer qué producto específico se compró, pero cualquier extracción
futura con más detalle de canasta debería mantener ese mismo nivel de
agregación.

## 5. Transparencia y explicabilidad

`shap.TreeExplainer` sobre LightGBM A2 permite explicar cualquier predicción
individual en una frase (ver `docs/manual_usuario.md`). Orden de importancia:
`recency_dias` > `umbral_propio_dias` > `n_dias_activo_12m` > `tendencia_gasto`.
El modelo **no** es una caja negra total: sus decisiones son auditable
feature por feature, aunque sigue siendo más complejo de explicar que una
regla fija (trade-off aceptado por la mejora de desempeño, documentada y
probada estadísticamente en `optimizacion.md`).

## 6. Impacto social

**Impacto positivo potencial:** permite dirigir esfuerzos de retención
(ofertas, contacto proactivo) hacia clientes en riesgo real, en vez de
campañas masivas indiscriminadas — mejor uso de presupuesto de marketing y,
en teoría, mejor experiencia para el cliente que recibe una oferta relevante
en el momento adecuado. **Impactos negativos potenciales:** una lista mal
comunicada podría usarse para prácticas agresivas de contacto (ver sección 7);
y, como se documenta en la sección 2, el diseño actual del umbral global puede
sub-atender a los clientes de alta frecuencia. **Quiénes se benefician:** el
negocio (menor pérdida de ingresos) y, si se usa bien, los propios clientes en
riesgo que reciben una oferta de retención oportuna. **Quiénes podrían verse
perjudicados:** clientes marcados como "en riesgo" por razones médicas ajenas
al negocio (sección 3), y clientes de alta frecuencia sub-representados en la
lista (sección 1) si se usa un presupuesto de campaña fijo sin ajustar por
segmento.

## 7. Responsabilidad

**¿Quién es responsable si el modelo falla?** El equipo de datos (Henry
Medina) es responsable de la calidad técnica del modelo y de documentar sus
limitaciones; el equipo de Marketing que opera la lista es responsable de
cómo se usa esa información en campañas reales (nunca de forma punitiva, ver
sección 8). Ninguna decisión de negocio debería tomarse de forma
**automática** a partir del score sin revisión humana, especialmente en el
segmento de comprador crónico (sección 3).

**Mecanismos de accountability:**
- Toda corrida de evaluación queda versionada en `results/metrics/` y
  `results/reports/`, con la fecha y el commit que la generó (una vez exista
  el repositorio de Git).
- El test (C5) se toca una sola vez (regla no negociable #2): ningún número
  de desempeño reportado pudo haber sido "elegido" ajustando el modelo
  después de verlo.
- Las pruebas de estrés (`optimizacion.md`) documentan explícitamente bajo
  qué condiciones el modelo deja de ser confiable, para que quien lo opere
  sepa cuándo desconfiar del resultado.

**Plan de monitoreo y actualización (propuesto, no implementado en este
alcance):**
1. Recalcular la tasa base de `churn` y el PR-AUC sobre cada corte nuevo
   (cada 120 días, siguiendo el mismo esquema temporal del proyecto) y
   compararlo contra el rango histórico (35%-42%, ver `analisis_datos.md`);
   una desviación mayor a 5 p.p. debería disparar una revisión, igual que el
   criterio ya usado para T2.1.
2. Monitorear el porcentaje de valores faltantes por feature en cada corrida
   de producción; más de 10% en cualquier feature numérica es la señal de
   alerta encontrada en las pruebas de estrés (`optimizacion.md`).
3. Reentrenar el modelo cuando se acumulen suficientes cortes nuevos (al
   menos uno adicional) para no perder el diseño out-of-time, nunca
   re-entrenando solo para "mejorar" un número sin evidencia de que el
   desempeño real se haya degradado.
4. Revisar la calibración (T3.5, pendiente) antes de usar el score como una
   probabilidad absoluta en cualquier reporte a negocio.

## 8. Uso dual y mal uso

El modelo produce un **ranking de riesgo de deserción**, no una decisión
automática. Está pensado para que Marketing priorice una lista limitada de
clientes a contactar (retención, promociones), no para negar servicio,
cambiar precios individualmente ni para ningún uso que perjudique al cliente
identificado como "en riesgo". Cualquier acción de negocio sobre la lista
debería ser positiva para el cliente (una oferta, un contacto de
seguimiento), nunca punitiva. **Salvaguarda implementada:** el modelo no
expone ni usa datos personales (sección 4), lo que limita el radio de daño si
la lista se filtrara o se usara fuera de su propósito original.

## 9. Limitaciones reconocidas

1. **Cobertura.** El modelo cubre ~26% de los clientes con actividad. El
   69,7% queda fuera por tener 1–3 compras — **no debe usarse** para decidir
   nada sobre esos clientes.
2. **Censura estructural.** 14,7%-15,8% global, 33% en el segmento bajo, 0%
   en el alto. Sesgo sistemático contra clientes lentos, no ruido aleatorio.
3. **Ventas anónimas.** 24,2% de las facturas sin cliente identificado. El
   modelo describe a los identificables.
4. **Deserción ≠ abandono comercial.** Ver sección 3.
5. **Sin variables de edad ni sexo.** Evita parte de la discusión de equidad
   demográfica directa, pero también impide medirla si existiera.
6. **`umbral_propio_dias` participa de la etiqueta.** Análisis de
   sensibilidad en `optimizacion.md`: el modelo no depende críticamente de
   ella (caída de solo 1,94% al quitarla).
7. **Privacidad.** Ver sección 4.
8. **Equidad entre segmentos.** Ver secciones 1 y 2: un umbral global
   sub-representa a los clientes de alta frecuencia en la lista de contacto.
9. **Calibración pendiente (T3.5).** Las probabilidades del modelo todavía no
   están calibradas. Usar el score directamente como "probabilidad de
   desertar" en una comunicación a negocio, antes de resolver T3.5, podría
   sobre- o subestimar el riesgo real en términos absolutos — el ranking (a
   quién priorizar) es confiable; el número exacto de probabilidad, todavía
   no. **No debe usarse** el score como una probabilidad calibrada hasta
   entonces.
10. **Muestra pequeña en comprador crónico.** Solo 630 clientes de test
    (0,9%) son comprador crónico. El PR-AUC reportado para ese grupo (0,75)
    debe leerse con cautela.
11. **Robustez limitada a datos muy degradados.** Las pruebas de estrés
    (`optimizacion.md`) muestran que el modelo pierde más del 10% de su
    PR-AUC con más de 30% de datos faltantes o con desbalances mucho más
    extremos que el real — **no debe confiarse en el modelo** si el pipeline
    de datos de producción presenta esos problemas sin que se detecten antes.
